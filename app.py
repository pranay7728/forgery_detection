from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import os
import tempfile
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)

# Initialize CORS
CORS(app)

# Load the TensorFlow model at startup for forgery detection
MODEL_PATH = 'forgeryvTestRand.keras'
try:
    forgery_model = tf.keras.models.load_model(MODEL_PATH)
    print("Forgery detection model loaded successfully.")
except Exception as e:
    print(f"Error loading forgery detection model: {e}")
    forgery_model = None

# Load the YOLOv8 model for object detection
yolo_model = YOLO('best2.pt')

def perform_ela(image_path, output_path, quality=90):
    try:
        # Open the original image
        original_image = Image.open(image_path)

        # Save the image at a lower quality (JPEG format)
        temp_image_path = 'temp_image.jpg'
        original_image.save(temp_image_path, 'JPEG', quality=quality)

        # Open the compressed image
        compressed_image = Image.open(temp_image_path)

        # Resize the compressed image to match the original if necessary
        if original_image.size != compressed_image.size:
            compressed_image = compressed_image.resize(original_image.size)

        # Ensure both images are in the same mode
        if original_image.mode != compressed_image.mode:
            compressed_image = compressed_image.convert(original_image.mode)

        # Perform error level analysis by finding the difference
        ela_image = ImageChops.difference(original_image, compressed_image)

        # Convert to RGB if the image is grayscale (single-channel)
        if ela_image.mode != 'RGB':
            ela_image = ela_image.convert('RGB')

        # Enhance the differences (make them more visible)
        extrema = ela_image.getextrema()
        max_diff = max([max(channel_extrema) for channel_extrema in extrema]) if isinstance(extrema[0], tuple) else max(extrema)
        scale = 255.0 / max_diff if max_diff != 0 else 1.0
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        # Convert the ELA image to numpy array for processing
        ela_array = np.array(ela_image)

        # Convert the ELA image to grayscale to analyze pixel intensities
        ela_gray = np.mean(ela_array, axis=2)

        # Save the grayscale ELA image
        ela_gray_image = Image.fromarray(ela_gray.astype(np.uint8))
        ela_gray_image.save(output_path)

        os.remove(temp_image_path)
        return output_path
    except Exception as e:
        print(f"Error in perform_ela: {e}")
        return None

def perform_forgery_inference(ela_image_path):
    try:
        # Load the ELA image
        img = cv2.imread(ela_image_path)

        # Resize the ELA image to the model input size
        resized = tf.image.resize(img, (256, 256))

        # Normalize the image and expand dimensions
        input_data = np.expand_dims(resized / 255.0, axis=0)

        # Perform inference using the forgery model
        yhat = forgery_model.predict(input_data)
        confidence_score = float(yhat[0][0])

        return confidence_score
    except Exception as e:
        print(f"Error in forgery inference: {e}")
        return None

def perform_yolo_inference(image_path):
    try:
        # Perform YOLO inference on the ELA image
        results = yolo_model.predict(source=image_path, save=False)

        return results
    except Exception as e:
        print(f"Error in YOLO inference: {e}")
        return None

def overlay_bounding_boxes(image_path, yolo_results):
    try:
        # Load the original image
        img = cv2.imread(image_path)

        # Get bounding box coordinates and class labels from YOLO results
        for result in yolo_results:
            boxes = result.boxes.xyxy  # xyxy format for bounding boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                # Draw the bounding box on the original image
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box

        # Save the result image with bounding boxes
        output_image_path = "output_image_with_boxes.jpg"
        cv2.imwrite(output_image_path, img)

        return output_image_path
    except Exception as e:
        print(f"Error in overlaying bounding boxes: {e}")
        return None

@app.route('/detect-forgery', methods=['POST'])
def process_image():
    if forgery_model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if file:
        try:
            # Secure the filename and save the uploaded image temporarily
            filename = secure_filename(file.filename)
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
                file_path = temp_file.name
                file.save(file_path)

            # Save the ELA image temporarily
            ela_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name

            # Perform ELA and save the result
            saved_ela_image_path = perform_ela(file_path, ela_output_path)
            if saved_ela_image_path is None:
                os.remove(file_path)
                return jsonify({'error': 'ELA processing failed.'}), 500

            # Perform inference on the saved ELA image for forgery detection
            confidence_score = perform_forgery_inference(saved_ela_image_path)
            if confidence_score is None:
                os.remove(file_path)
                os.remove(saved_ela_image_path)
                return jsonify({'error': 'Inference failed.'}), 500

            # Determine if forgery is detected
            if confidence_score < 0.5:
                # Perform YOLO inference if forgery is detected
                yolo_results = perform_yolo_inference(saved_ela_image_path)
                if yolo_results is None:
                    return jsonify({'error': 'YOLO inference failed.'}), 500

                # Overlay bounding boxes on the original image
                output_image_with_boxes = overlay_bounding_boxes(file_path, yolo_results)
                if output_image_with_boxes is None:
                    return jsonify({'error': 'Error overlaying bounding boxes.'}), 500

                # Return the result along with the bounding box image
                response = {
                    'forgery_detected': True,
                    'confidence': round(1 - confidence_score, 4),
                    'output_image': output_image_with_boxes
                }
            else:
                response = {
                    'forgery_detected': False,
                    'confidence': round(confidence_score, 4)
                }

            # Clean up temporary files
            os.remove(file_path)
            os.remove(saved_ela_image_path)

            return jsonify(response), 200

        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': 'Internal server error.'}), 500

    return jsonify({'error': 'Invalid request.'}), 400

@app.route('/')
def index():
    return """
    <h1>Image Forgery Detection and YOLO API</h1>
    <p>Use the <code>/detect-forgery</code> endpoint to upload an image and get predictions.</p>
    <p>Example using <code>curl</code>:</p>
    <pre>
    curl -X POST -F "image=@path_to_image.jpg" http://localhost:5000/detect-forgery
    </pre>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
