from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib

app = Flask(__name__)

# Load model and scaler
model = tf.keras.models.load_model("credit_risk_pred.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return "Credit Risk Prediction API is up!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prob = model.predict(features_scaled)[0][0]
        prediction = int(prob >= 0.5)

        return jsonify({
            'prediction': prediction,
            'probability': round(float(prob), 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
