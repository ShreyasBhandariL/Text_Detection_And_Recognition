from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import easyocr
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads/"

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

# Initialize EasyOCR Reader without GPU
reader = easyocr.Reader(["en"], gpu=False)


def preprocess_image(image_path):
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to handle different lighting conditions
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    return binary


def extract_text_from_image(image_path):
    preprocessed_img = preprocess_image(image_path)
    pil_img = Image.fromarray(preprocessed_img)

    # Use EasyOCR to extract text
    result = reader.readtext(np.array(pil_img))
    text = " ".join([res[1] for res in result])
    return text


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
