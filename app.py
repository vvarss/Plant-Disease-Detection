
import logging
logging.basicConfig(level=logging.DEBUG)

import warnings
warnings.filterwarnings("ignore")


# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# import base64
# import io
# from PIL import Image
# #from flask_cors import CORS


# from flask import Flask, request, jsonify, make_response

# app = Flask(__name__)

# @app.after_request
# def add_cors_headers(response):
#     response.headers["Access-Control-Allow-Origin"] = "*"
#     response.headers["Access-Control-Allow-Headers"] = "Content-Type"
#     response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
#     return response

# model = tf.keras.models.load_model('model.h5')

# # def preprocess_image(image_data):
# #     image = Image.open(io.BytesIO(image_data)).convert("RGB")
# #     image = image.resize((224, 224))  # or your model's expected size
# #     image = np.array(image) / 255.0
# #     return np.expand_dims(image, axis=0)
# # print(model.summary())

# from PIL import Image
# import numpy as np
# import io

# def preprocess_image(image_data):
#     image = Image.open(io.BytesIO(image_data)).convert("RGB")  # Open the image and ensure it's in RGB format
#     image = image.resize((62, 62))  # Resize to 62x62 to match the model input size
    
#     image = np.array(image)  # Convert to numpy array
#     image = image / 255.0  # Normalize pixel values to [0, 1]
    
#     return np.expand_dims(image, axis=0)  # Add batch dimension (shape becomes (1, 62, 62, 3))

# # Use the function before making the prediction
# processed = preprocess_image(image_data)  # Assuming image_data is the raw image byte data
# prediction = model.predict(processed)


# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     if 'image' not in data:
#         return jsonify({'error': 'No image provided'})

#     # Decode base64
#     image_data = base64.b64decode(data['image'].split(',')[1])
#     processed = preprocess_image(image_data)

#     prediction = model.predict(processed)
#     predicted_class = np.argmax(prediction)

#     return jsonify({'prediction': str(predicted_class)})

# if __name__ == '__main__':
#     app.run(debug=True)


# import logging
# from flask import Flask, request, jsonify
# import tensorflow as tf
# import numpy as np
# import io
# from PIL import Image

# # Set logging level to DEBUG
# logging.basicConfig(level=logging.DEBUG)

# app = Flask(__name__)

# # Allow cross-origin requests (CORS)
# @app.after_request
# def add_cors_headers(response):
#     response.headers["Access-Control-Allow-Origin"] = "*"
#     response.headers["Access-Control-Allow-Headers"] = "Content-Type"
#     response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
#     return response

# # Load model once when the app starts
# try:
#     model = tf.keras.models.load_model('model.h5')
#     logging.debug(f"Model loaded successfully.")
# except Exception as e:
#     logging.error(f"Error loading model: {e}")

# # Define classes globally
# classes = ["healthy", "early blight", "late blight"]

# def preprocess_image(image_data):
#     try:
#         image = Image.open(io.BytesIO(image_data)).convert("RGB")  # Ensure the image is RGB
#         image = image.resize((128, 128))  # Resize to (128, 128) to match the model input size
#         image = np.array(image)  # Convert to a numpy array
#         image = image / 255.0  # Normalize the pixel values to [0, 1]
#         return np.expand_dims(image, axis=0)  # Add batch dimension (shape becomes (1, 128, 128, 3))
#     except Exception as e:
#         logging.error(f"Error in preprocess_image: {e}")
#         raise

# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             # Read image file
#             if 'image' not in request.files:
#                 raise ValueError("No image file found in request")
                
#             image_data = request.files['image'].read()
#             processed_image = preprocess_image(image_data)  # Process image

#             logging.debug(f"Image shape before prediction: {processed_image.shape}")

#             # Model prediction
#             prediction = model.predict(processed_image)

#             logging.debug(f"Raw prediction: {prediction}")

#             # Apply argmax to get the class with the highest probability
#             predicted_class_index = np.argmax(prediction)
#             logging.debug(f"Predicted class index: {predicted_class_index}")

#             # Get the class name based on the predicted index
#             predicted_class = classes[predicted_class_index]
#             logging.debug(f"Predicted class: {predicted_class}")

#             # Return the predicted class as the response
#             return jsonify({'prediction': predicted_class})

#         except Exception as e:
#             logging.error(f"Error during prediction: {e}")
#             return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# import tensorflow as tf
# import numpy as np
# import io
# from PIL import Image

# app = Flask(__name__)

# # Load the trained model
# model = tf.keras.models.load_model('model.h5')

# # Define the classes for tomato detection
# classes = ["healthy", "early blight", "late blight"]

# # Preprocess the uploaded image to match the model's input format
# def preprocess_image(image_data):
#     image = Image.open(io.BytesIO(image_data)).convert("RGB")  # Ensure the image is RGB
#     image = image.resize((128, 128))  # Resize to (128, 128) to match the model input size
#     image = np.array(image)  # Convert to a numpy array
#     image = image / 255.0  # Normalize the pixel values to [0, 1]
#     return np.expand_dims(image, axis=0)  # Add batch dimension (shape becomes (1, 128, 128, 3))

# # Home page route
# @app.route('/')
# def index():
#     # You can add any additional logic here if you want to show results on the main page
#     return render_template('index.html')

# # Prediction route for uploading an image
# @app.route('/predict', methods=['POST'])
# def predict():
#     if request.method == 'POST':
#         try:
#             image_data = request.files['image'].read()  # Read image file
#             processed_image = preprocess_image(image_data)  # Process the image

#             # Make the prediction
#             prediction = model.predict(processed_image)

#             # Get the predicted class index
#             predicted_class_index = np.argmax(prediction)

#             # Get the class label based on the index
#             predicted_class = classes[predicted_class_index]

#             # Return the prediction result as a JSON response
#             return jsonify({'prediction': predicted_class})

#         except Exception as e:
#             return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     # Run the Flask application
#     app.run(debug=True)

from flask import Flask, request, jsonify
from PIL import Image
import io
import base64
import numpy as np
import tensorflow as tf

# Initialize the Flask application
app = Flask(__name__)

# Load your pre-trained model (adjust the path as necessary)
model = tf.keras.models.load_model('model.h5')

def prepare_image(image_data):
    # Decode the image from base64
    img_data = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(img_data))

    # Resize the image to the size expected by the model
    img = img.resize((224, 224))  # Adjust the size according to your model's input size
    img = np.array(img)

    # Normalize the image (this depends on how your model was trained)
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({'error': 'No image data found'}), 400

    # Prepare the image
    image_data = data['image']
    img = prepare_image(image_data)

    # Predict the class of the image
    prediction = model.predict(img)
    class_idx = np.argmax(prediction, axis=1)[0]

    # Map the prediction to your class labels
    class_labels = ['Healthy', 'Early Blight', 'Late Blight']
    predicted_class = class_labels[class_idx]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
