# ============================================
# app.py - Web Interface using Flask
# Run: python app.py
# Then open: http://localhost:5000
# ============================================

import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max

MODEL_PATH = "model/bdt_detector.h5"
IMG_SIZE = (128, 128)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

model = load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        return "REAL", round(float(prediction) * 100, 2)
    else:
        return "FAKE", round(float(1 - prediction) * 100, 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    label, confidence = predict(filepath)

    return jsonify({
        'label': label,
        'confidence': confidence,
        'image_url': f'/static/uploads/{filename}'
    })

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)
