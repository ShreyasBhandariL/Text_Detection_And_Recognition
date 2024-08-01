from flask import Flask, request, render_template, redirect
from PIL import Image
import easyocr
import cv2
import numpy as np
import os
import difflib

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads/"

# Ensure the upload folder exists
if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Initialize EasyOCR Reader without GPU
reader = easyocr.Reader(["en"], gpu=False)


def preprocess_image(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    contrast = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(contrast, (3, 3), 0)

    # Use Otsu's thresholding for binarization
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological operations
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.dilate(binary, kernel, iterations=1)
    processed = cv2.erode(processed, kernel, iterations=1)

    return processed


def correct_text(text):
    # Define a dictionary of common OCR errors and their corrections
    corrections = {
        "Poat": "Post",
        "Teum": "Team",
        "bui": "build",
        "Idip": "building",
        "hud": "had",
        "Wedisd y": "Wednesday",
        "foad": "food",
    }

    # Split the text into words
    words = text.split()

    # Correct each word
    corrected_words = []
    for word in words:
        closest_match = difflib.get_close_matches(word, corrections.keys(), n=1)
        if closest_match:
            corrected_words.append(corrections[closest_match[0]])
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)


def extract_text_from_image(image_path):
    # Preprocess the image
    preprocessed_img = preprocess_image(image_path)
    pil_img = Image.fromarray(preprocessed_img)

    # Use EasyOCR to extract text
    result = reader.readtext(np.array(pil_img), detail=0, paragraph=True)
    text = " ".join(result)

    # Correct common OCR errors
    corrected_text = correct_text(text)
    return corrected_text


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)
            extracted_text = extract_text_from_image(filepath)
            return render_template("index.html", text=extracted_text)
    return render_template("index.html", text="")


if __name__ == "__main__":
    app.run(debug=True)
