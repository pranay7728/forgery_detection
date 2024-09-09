from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('forgery_detect.keras')

# Function to preprocess the image (resize and normalize)
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Load the image
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

    # Save the uploaded image
    filename = secure_filename(image.filename)
    filepath = os.path.join('uploads', filename)
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

# Run the Flask app
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
