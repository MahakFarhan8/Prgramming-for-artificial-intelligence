from flask import Flask, render_template, request, send_from_directory
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)

# Define folders for uploads and results
UPLOAD_FOLDER = "main/uploads"
RESULT_FOLDER = "main/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Using YOLOv8 Nano (fast & lightweight)

# List of COCO animal class IDs
ANIMAL_CLASSES = [16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No file selected", 400

    # Save the uploaded image
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    result_path = os.path.join(RESULT_FOLDER, "result_" + file.filename)
    file.save(file_path)

    # Perform detection
    results = model(file_path)
    boxes = results[0].boxes
    filtered_boxes = []

    # Filter only animal detections
    for box in boxes:
        cls_id = int(box.cls)  # Class ID of detected object
        if cls_id in ANIMAL_CLASSES:
            filtered_boxes.append(box)

    # If no animals were detected, return a message
    if not filtered_boxes:
        return "No animal herd detected.", 200

    # Save filtered detection results
    results[0].save(filename=result_path)

    return render_template("result.html", filename="result_" + file.filename)

@app.route("/static/results/<filename>")
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)

