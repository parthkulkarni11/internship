from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('disease_model.pkl')
le_crop = joblib.load('crop_encoder.pkl')
le_soil = joblib.load('soil_encoder.pkl')
le_disease = joblib.load('disease_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    crop = request.form['crop']
    soil = request.form['soil']
    symptom = request.form['symptom']  # Optional

    try:
        # Encode crop and soil
        crop_encoded = le_crop.transform([crop])[0]
        soil_encoded = le_soil.transform([soil])[0]

        # Predict using model
        features = np.array([[crop_encoded, soil_encoded]])
        prediction_encoded = model.predict(features)[0]
        predicted_disease = le_disease.inverse_transform([prediction_encoded])[0]

        # Show symptom if given
        if symptom:
            result = f'Predicted Disease: {predicted_disease} | Reported Symptom: {symptom}'
        else:
            result = f'Predicted Disease: {predicted_disease}'

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text='Invalid input! Please check selections.')

if __name__ == '__main__':
    app.run(debug=True)
