from flask import Flask, request, render_template
import numpy as np
import cv2
import pandas as pd
import base64
from Alphabet_Recognition import AlphabetRecognizer 

app = Flask(__name__)

# Instantiate your AlphabetRecognizer class
alphabet_recognizer = AlphabetRecognizer(dataset_path="dataset_edited/")
alphabet_recognizer.load_dataset()
alphabet_recognizer.split_dataset()
alphabet_recognizer.preprocess_data()
alphabet_recognizer.build_model()
alphabet_recognizer.compile_model()
alphabet_recognizer.train_model()

@app.route('/')
def index():
    return render_template("main.html")

@app.route('/predict', methods=["POST"])
def get_image():
    # Retrieve image data from the canvas
    canvasdata = request.form['canvasimg']
    encoded_data = request.form['canvasimg'].split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess the image
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (28, 28), interpolation=cv2.INTER_LINEAR)
    gray_image = gray_image / 255.0
    gray_image = np.expand_dims(gray_image, axis=-1)
    img = np.expand_dims(gray_image, axis=0)

    # Make prediction using the loaded model
    prediction = alphabet_recognizer.model.predict(img)
    predicted_class = np.argmax(prediction)
    predicted_letter = alphabet_recognizer.class_to_letter[predicted_class]
    
    return render_template("main.html", value=predicted_letter)

if __name__ == '__main__':
    app.run(debug=True)
