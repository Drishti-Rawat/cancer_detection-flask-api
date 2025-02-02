
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

CORS(app, origins=["http://localhost:3000"])
# Load your models
cancer_model = joblib.load('cancer_model.sav')
skin_cancer_model = joblib.load('skin_cancer.sav')
lung_cancer_model = joblib.load('Lung_cancer.sav')

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
    

def preprocess_lung_cancer_data(data):
    """
    Preprocess lung cancer input data:
    - Gender should be 'M' or 'F' (will be converted to 1/2 internally)
    - All other features (except age) should be 1 (No) or 2 (Yes)
    """
    # Define the exact order of features as per the dataset
    feature_order = [
        'gender', 'age', 'smoking', 'yellow_fingers', 'anxiety',
        'peer_pressure', 'chronic_disease', 'fatigue', 'allergy', 'wheezing',
        'alcohol', 'coughing', 'shortness_of_breath', 'swallowing_difficulty',
        'chest_pain'
    ]
    
    # Input validation
    for key, value in data.items():
        if key == 'gender':
            if value not in ['M', 'F']:
                raise ValueError(f"Gender must be 'M' or 'F'. Got {value}")
        elif key != 'age':
            if str(value) not in ['1', '2']:
                raise ValueError(f"Feature '{key}' must have value 1 (No) or 2 (Yes). Got {value}")
    
    processed_data = []
    for feature in feature_order:
        if feature == 'age':
            processed_data.append(float(data[feature]))
        elif feature == 'gender':
            # Convert M/F to 1/2
            gender_value = 2 if data[feature] == 'M' else 1
            processed_data.append(gender_value)
        else:
            # Convert string numbers to integers
            processed_data.append(int(data[feature]))
    
    return np.array(processed_data).reshape(1, -1)

@app.route('/predict/lung_cancer', methods=['POST'])
def predict_lung_cancer():
    try:
        # Get data from request
        data = request.json
        
        # Preprocess the data
        processed_data = preprocess_lung_cancer_data(data)
        
        # Make prediction
        prediction = lung_cancer_model.predict(processed_data)
        
        # Convert prediction to human-readable format
        result = "YES" if prediction[0] == 1 else "NO"
        
        return jsonify({
            'success': True,
            'prediction': result,
            # 'prediciton_result' : prediction[0],S
            'probability': prediction.tolist()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

    
if __name__ == '__main__':
    app.run(debug=True)