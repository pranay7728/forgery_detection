from flask import Flask, request, jsonify
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import os
import tempfile
import cv2
import tensorflow as tf

app = Flask(__name__)

# Load the TensorFlow model at startup
MODEL_PATH = 'forgeryvTestRand.keras'  # Ensure this path is correct
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def perform_ela(image_path, quality=90):
    """
    Performs Error Level Analysis (ELA) on the given image.

    Args:
        image_path (str): Path to the original image.
        quality (int): JPEG quality for compression.

    Returns:
        np.ndarray: Grayscale ELA image as a NumPy array.
    """
    try:
        # Open the original image
        original_image = Image.open(image_path).convert('RGB')

        # Save the image at a lower quality (JPEG format)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_image_path = temp_file.name
        original_image.save(temp_image_path, 'JPEG', quality=quality)

        # Open the compressed image
        compressed_image = Image.open(temp_image_path).convert('RGB')

        # Perform error level analysis by finding the difference
        ela_image = ImageChops.difference(original_image, compressed_image)

        # Enhance the differences (make them more visible)
        extrema = ela_image.getextrema()
        max_diff = max([max(channel_extrema) for channel_extrema in extrema])
        scale = 255.0 / max_diff if max_diff != 0 else 1.0
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

        # Convert the ELA image to numpy array for processing
        ela_array = np.array(ela_image)

        # Convert the ELA image to grayscale to analyze pixel intensities
        ela_gray = np.mean(ela_array, axis=2).astype(np.uint8)  # Shape: (H, W)

        # Clean up the temporary compressed image
        os.remove(temp_image_path)

        return ela_gray
    except Exception as e:
        print(f"Error in perform_ela: {e}")
        return None

def perform_inference(ela_gray):
    """
    Performs inference on the ELA-processed grayscale image.

    Args:
        ela_gray (np.ndarray): Grayscale ELA image.

    Returns:
        float: Confidence score from the model prediction.
    """
    try:
        # Resize to 256x256 as per the model's requirement
        resized = cv2.resize(ela_gray, (256, 256))

        # Stack to make 3 channels if model expects RGB
        resized = np.stack([resized]*3, axis=-1)  # Shape: (256, 256, 3)

        # Normalize the image
        resized = resized.astype(np.float32) / 255.0

        # Expand dimensions to match model input (1, 256, 256, 3)
        input_data = np.expand_dims(resized, axis=0)

        # Predict
        yhat = model.predict(input_data)
        confidence_score = float(yhat[0][0])

        return confidence_score
    except Exception as e:
        print(f"Error in perform_inference: {e}")
        return None

@app.route('/detect-forgery', methods=['POST'])
def process_image():
    """
    Endpoint to process an uploaded image, perform ELA, run inference, and return the result.

    Returns:
        JSON: {
            'forgery_detected': True or False,
            'confidence': float
        }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image part in the request.'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if file:
        try:
            # Save the uploaded image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
                file_path = temp_file.name
                file.save(file_path)

            # Perform ELA
            ela_gray = perform_ela(file_path)
            if ela_gray is None:
                os.remove(file_path)
                return jsonify({'error': 'ELA processing failed.'}), 500

            # Perform inference
            confidence_score = perform_inference(ela_gray)
            if confidence_score is None:
                os.remove(file_path)
                return jsonify({'error': 'Inference failed.'}), 500

            # Determine if forgery is detected based on confidence score
            if confidence_score < 0.5:
                response = {
                    'forgery_detected': True,
                    'confidence': round(1 - confidence_score, 4)
                }
            else:
                response = {
                    'forgery_detected': False,
                    'confidence': round(confidence_score, 4)
                }

            # Clean up the uploaded file
            os.remove(file_path)

            return jsonify(response), 200

        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({'error': 'Internal server error.'}), 500

    return jsonify({'error': 'Invalid request.'}), 400

@app.route('/')
def index():
    return """
    <h1>Image Forgery Detection API</h1>
    <p>Use the <code>/process</code> endpoint to upload an image and get predictions.</p>
    <p>Example using <code>curl</code>:</p>
    <pre>
    curl -X POST -F "image=@path_to_image.jpg" http://localhost:5000/process
    </pre>
    """

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
