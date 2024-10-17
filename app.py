from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load your trained Random Forest model
model = joblib.load('cancer_model.sav')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
      prediction = model.predict([data['features']])
      return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 
