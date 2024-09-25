from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageChops, ImageEnhance
import io

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes (or restrict to specific origins)
CORS(app)

# Define the upload directory using an absolute path
UPLOAD_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Load the trained model
model = tf.keras.models.load_model('forgeryvTestRand.keras')

# Updated ELA function to match the first implementation
def perform_ela(image_path, quality=90):
    # Open the original image
    original_image = Image.open(image_path)

    # Save the image at a lower quality (JPEG format)
    temp_image_path = 'temp_image.jpg'
    original_image.save(temp_image_path, 'JPEG', quality=quality)

    # Open the compressed image
    compressed_image = Image.open(temp_image_path)

    # Resize and mode conversion checks
    if original_image.size != compressed_image.size:
        compressed_image = compressed_image.resize(original_image.size)
    if original_image.mode != compressed_image.mode:
        compressed_image = compressed_image.convert(original_image.mode)

    # Perform error level analysis by finding the difference
    ela_image = ImageChops.difference(original_image, compressed_image)

    # Convert to RGB if the image is grayscale (single-channel)
    if ela_image.mode != 'RGB':
        ela_image = ela_image.convert('RGB')

    # Enhance the differences (make them more visible)
    extrema = ela_image.getextrema()

    # Extract extrema from each channel if necessary
    if isinstance(extrema[0], tuple):
        max_diff = max([max(channel_extrema) for channel_extrema in extrema])
    else:
        max_diff = max(extrema)

    # Prevent division by zero if max_diff is 0
    scale = 255.0 / max_diff if max_diff != 0 else 1.0
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)

    # Convert the ELA image to numpy array for processing
    ela_array = np.array(ela_image)

    # Convert the ELA image to grayscale to analyze pixel intensities
    ela_gray = np.mean(ela_array, axis=2)  # Average the RGB channels to get grayscale intensity

    # Convert back to PIL Image
    ela_gray_image = Image.fromarray(ela_gray.astype(np.uint8))

    # Clean up temporary file
    os.remove(temp_image_path)

    return ela_gray_image

# Function to preprocess the image (apply ELA, resize, normalize)
def preprocess_image(image_path):
    # Perform ELA on the image
    ela_image = perform_ela(image_path)

    # Convert the ELA image to a numpy array
    ela_array = np.array(ela_image)

    # Resize the ELA image to match the input shape (256x256)
    ela_resized = cv2.resize(ela_array, (256, 256))

    # Normalize pixel values (0-1)
    ela_resized = ela_resized / 255.0

    # Add batch dimension and channel dimension (for grayscale)
    ela_resized = np.expand_dims(ela_resized, axis=(0, -1))

    return ela_resized

# Function to make prediction
def predict_image(image_path):
    img = preprocess_image(image_path)  # Preprocess the image (with ELA)
    prediction = model.predict(img)  # Get prediction
    return prediction[0][0]  # Return the probability (since sigmoid is used)

# Flask route for forgery detection
@app.route('/detect-forgery', methods=['POST'])
def detect_forgery():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    # Get the file from the request
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Save the uploaded image
        filename = secure_filename(image.filename)
        filepath = os.path.join(UPLOAD_DIRECTORY, filename)
        image.save(filepath)

        # Make a prediction
        prediction = predict_image(filepath)

        # Remove the uploaded image after prediction
        os.remove(filepath)

        # Return the result as JSON
        if prediction < 0.5:
            return jsonify({
                'forgery_detected': True,
                'confidence': float(1 - prediction)
            })
        else:
            return jsonify({
                'forgery_detected': False,
                'confidence': float(prediction)
            })

    except Exception as e:
        # Log the error (you might want to use a proper logging system in production)
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the image'}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)