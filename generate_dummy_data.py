# ============================================
# generate_dummy_data.py
# Creates fake dataset images for testing
# Run this ONLY if you don't have real images
# ============================================

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random

REAL_DIR = "dataset/real"
FAKE_DIR = "dataset/fake"
NUM_IMAGES = 100  # per class

os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(FAKE_DIR, exist_ok=True)

def generate_note(path, is_real=True):
    img = Image.new("RGB", (300, 150), color=(
        random.randint(180, 220),
        random.randint(140, 180),
        random.randint(60, 100)
    ))
    draw = ImageDraw.Draw(img)

    # Draw some shapes to simulate a banknote
    for _ in range(8):
        x1, y1 = random.randint(0, 250), random.randint(0, 100)
        x2, y2 = x1 + random.randint(10, 50), y1 + random.randint(10, 50)
        color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    # Add text
    draw.text((10, 10), "Bangladesh Bank", fill=(30, 30, 30))
    draw.text((10, 40), "১০০ টাকা", fill=(30, 30, 30))

    if is_real:
        # Real notes have a "security line"
        draw.line([(0, 75), (300, 75)], fill=(200, 150, 50), width=3)
        draw.text((200, 120), "REAL", fill=(0, 120, 0))
    else:
        # Fake notes are slightly distorted / blurry look
        draw.text((200, 120), "COPY", fill=(150, 0, 0))
        for _ in range(5):
            x = random.randint(0, 300)
            draw.line([(x, 0), (x + 5, 150)], fill=(100, 100, 100, 80), width=1)

    img.save(path)

print("Generating dummy dataset...")
for i in range(NUM_IMAGES):
    generate_note(f"{REAL_DIR}/real_{i}.jpg", is_real=True)
    generate_note(f"{FAKE_DIR}/fake_{i}.jpg", is_real=False)

print(f" Done! {NUM_IMAGES} real + {NUM_IMAGES} fake images created in dataset/")
print(" For real results, replace these with actual BDT currency photos!")
