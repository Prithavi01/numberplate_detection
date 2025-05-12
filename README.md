Number Plate Detection and OCR with EasyOCR
This repository provides a solution for detecting and recognizing number plates in images using OpenCV and EasyOCR.

The main functionality of the code includes:

Detecting number plates in images using a Haar Cascade classifier.

Extracting text from the detected number plate using EasyOCR.

Displaying the detected number plate and the extracted text on the image.

Requirements
To run the code, you need the following Python libraries:

opencv-python for image processing and object detection.

easyocr for Optical Character Recognition (OCR).

numpy for numerical operations.

google.colab (only if using Google Colab for displaying images).

Installation
Install the required dependencies by running:

pip install opencv-python easyocr numpy
Usage
1. Prepare your image
Ensure that you have an image of a vehicle with a clear view of the number plate.

2. Script to Detect Number Plates
You can run the provided Python script that loads an image, detects the number plate, and extracts the text using EasyOCR. Below is the code to use:

import cv2
import easyocr
import numpy as np
from google.colab.patches import cv2_imshow  # For image display in Colab

def detect_number_plate(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error loading image!")
        return

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load Haar Cascade for number plate detection
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')

    # Detect plates in the image
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Load EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Loop through detected plates
    for (x, y, w, h) in plates:
        # Draw a rectangle around the detected plate
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the plate area
        plate_img = img[y:y+h, x:x+w]

        # Perform OCR on the cropped plate image
        result = reader.readtext(plate_img)

        # Display detected text on the original image
        for detection in result:
            text = detection[1]
            print(f"Detected Number Plate: {text}")
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the final image with annotations
    cv2_imshow(img)  # Works in Colab only

# Run the function
image_path = 'path_to_your_image.jpg'  # Change this to your image path
detect_number_plate(image_path)
3. Input Image
Replace image_path = 'path_to_your_image.jpg' with the path of your image containing the number plate.

4. Output
The script will:

Detect number plates in the image.

Extract the number plate text using EasyOCR.

Display the image with bounding boxes around detected plates and the detected text overlaid.

Notes
The Haar Cascade used here is specifically trained for Russian number plates (haarcascade_russian_plate_number.xml), but you can experiment with different Haar cascades depending on the region or country.

The detection may not work well on images with low resolution or where the number plate is obscured.

Troubleshooting
If the image does not load, check the image path and make sure the file exists.

If OCR does not detect the number plate correctly, try increasing the resolution of the input image or fine-tuning the cascade parameters.

License
This project is licensed under the MIT License - see the LICENSE file for details.
