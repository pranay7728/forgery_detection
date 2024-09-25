from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

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

# Function to preprocess the image (resize and normalize)
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Load the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))  # Resize to match the input shape (256x256)
    img = img / 255.0  # Normalize pixel values (0-1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to make prediction
def predict_image(image_path):
    img = preprocess_image(image_path)  # Preprocess the image
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
