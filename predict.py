# ============================================
# predict.py - Test a single currency image
# Usage: python predict.py --image path/to/note.jpg
# ============================================

import numpy as np
import argparse
from keras.models import load_model
from keras.preprocessing import image

MODEL_PATH = "model/bdt_detector.h5"
IMG_SIZE = (128, 128)

def predict_currency(img_path):
    model = load_model(MODEL_PATH)

    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        label = "✅ REAL Currency"
        confidence = prediction * 100
    else:
        label = "❌ FAKE Currency"
        confidence = (1 - prediction) * 100

    print(f"\nResult: {label}")
    print(f"Confidence: {confidence:.2f}%")
    return label, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to currency image")
    args = parser.parse_args()
    predict_currency(args.image)
