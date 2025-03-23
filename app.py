from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('brain_tumor_vgg16_model1.h5')

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
#

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    img_file = request.files['file']
    img_path = os.path.join('static/uploads', img_file.filename)
    img_file.save(img_path)

    # Preprocess image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    result = CLASS_NAMES[class_index]

    return render_template('index.html', prediction=result, img_path=img_path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # Render uses port 8080 by default

