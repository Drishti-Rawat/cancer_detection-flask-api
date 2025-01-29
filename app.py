
import traceback
from flask import Flask, request, jsonify
import joblib
import pickle
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import cv2
import base64

app = Flask(__name__)
CORS(app)

# Load your models
cancer_model = joblib.load('cancer_model.sav')
skin_cancer_model = joblib.load('skin_cancer.sav')

def preprocess_image(image_data):
    # Decode base64 image
    img_bytes = base64.b64decode(image_data.split(',')[1])
    img = Image.open(io.BytesIO(img_bytes))
    
    # Resize image to required dimensions (64x64)
    img = img.resize((64, 64))
    
    # Convert to numpy array and normalize
    img_array = np.array(img)
    img_array = img_array / 255.0
    
    # Ensure shape is (1, 64, 64, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


@app.route('/', methods=['GET'])
def home():
    return "Cancer Detection API is running!"

@app.route('/predict/general_cancer', methods=['POST'])
def predict_general_cancer():
    data = request.json
    try:
        prediction = cancer_model.predict([data['features']])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

@app.route('/predict/skin_cancer', methods=['POST'])
def predict_skin_cancer():
    try:
        # Get image data from request
        image_data = request.json['image']
        
        # Preprocess the image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = skin_cancer_model.predict(processed_image)
        
        # Convert prediction to list for JSON serialization
        prediction_list = prediction.tolist()
        
        return jsonify({
            'success': True,
            'prediction': prediction_list
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
    
if __name__ == '__main__':
    app.run(debug=True)