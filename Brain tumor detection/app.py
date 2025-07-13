from flask import Flask, request, render_template
from keras.models import load_model
import cv2
import numpy as np

import os
import sys, os
print("✅ Running from:", sys.executable)
print("📁 Template path:", os.path.join(os.path.dirname(__file__), 'templates'))
print("📄 Templates found:", os.listdir(os.path.join(os.path.dirname(__file__), 'templates')))

app = Flask(__name__, template_folder=os.path.join(os.getcwd(), 'templates'))
print("✅ Running app.py from DL project2 folder")

try:
    model = load_model('brain_tumor_model.h5')
except Exception as e:
    print("Error loading the model:", e)
    raise

from flask import Flask, render_template

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = ''
    if request.method == 'POST':
        file = request.files['image']
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (100, 100)).reshape(1, 100, 100, 1) / 255.0
        pred = model.predict(img)
        prediction = 'Tumor Detected' if np.argmax(pred) == 1 else 'No Tumor Detected'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

