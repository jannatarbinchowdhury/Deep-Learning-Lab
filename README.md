# 🇧🇩 Bangladesh Taka Fake Currency Detector
A simple deep learning project using CNN (Convolutional Neural Network) to detect fake Bangladesh Taka notes.

---

##  Project Structure

```
bdt_fake_detector/
│
├── dataset/
│   ├── real/          ← Put REAL currency images here (JPG/PNG)
│   └── fake/          ← Put FAKE currency images here (JPG/PNG)
│
├── model/             ← Trained model will be saved here
├── static/uploads/    ← Web app saves uploaded images here
├── templates/
│   └── index.html     ← Web interface
│
├── train_model.py          ← Step 2: Train the model
├── predict.py              ← Step 3a: Test single image (terminal)
├── app.py                  ← Step 3b: Web interface
├── generate_dummy_data.py  ← (Optional) Generate test images
└── requirements.txt        ← All dependencies
```

---

##  What You Need

| Tool       | Version   |
|------------|-----------|
| Python     | 3.9 - 3.11 |
| pip        | Latest    |
| RAM        | 4GB+      |
| GPU        | Optional (CPU works fine) |

---

## How to Run (Step by Step)

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Prepare your dataset

**Option A (Real dataset — Recommended):**
- Collect 100–500 photos of REAL BDT notes → put in `dataset/real/`
- Collect 100–500 photos of FAKE BDT notes → put in `dataset/fake/`

**Option B (Dummy data — Just for testing the code):**
```bash
python generate_dummy_data.py
```
> Dummy data won't give accurate real-world results!

### Step 3 — Train the model
```bash
python train_model.py
```
- This will train for 20 epochs
- Best model saved to `model/bdt_detector.h5`
- Training chart saved to `model/training_result.png`

### Step 4 — Test with a single image
```bash
python predict.py --image path/to/your/note.jpg
```

### Step 5 — Launch Web App
```bash
python app.py
```
Then open your browser: **http://localhost:5000**

---

## How It Works

```
Currency Image
      ↓
  Resize to 128x128
      ↓
  CNN Model (3 Conv layers)
      ↓
  Output: REAL or FAKE + Confidence %
```

---

##  Tips for Better Accuracy

- Use at least **200+ images per class** for training
- Take photos under **good lighting**
- Include different **denominations**: ৳10, ৳20, ৳50, ৳100, ৳500, ৳1000
- Take photos from **multiple angles**
- The more data, the better the model!

---

##  Disclaimer
This project is for educational purposes only.
