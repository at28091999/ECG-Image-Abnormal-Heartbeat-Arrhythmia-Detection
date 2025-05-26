from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("ecg_cnn_model.h5") 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded."
    file = request.files['file']
    if file.filename == '':
        return "No selected file."
    
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    result = model.predict(img_array)
    os.remove(filepath) 

    if result[0][0] < 0.5:
        prediction = "Normal Heartbeat, No Risk"
    else:
        prediction = "Abnormal Heartbeat, Arrhythmia Detected, Risk of Heart Attack"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True)
