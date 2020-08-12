import base64
import time
import traceback
from io import BytesIO

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image

from model import MNISTClassifier

mnist_model = MNISTClassifier(model_path="weights/pretrained_mnist.pb")
app = Flask(__name__)
CORS(app)


def read_image(img_bytes):
    return cv2.imdecode(np.asarray(bytearray(img_bytes.read()), dtype="uint8"), cv2.IMREAD_COLOR)


@app.route('/predict', methods=['POST'])
def mnist_predict():
    start_time = time.time()
    data = {}
    if request.method == "POST":
        image = request.files.get("image", None)
        if image is not None:
            try:
                image = read_image(image)
                data = mnist_model.predict(image)
            except Exception as ex:
                print(traceback.format_exc())
                data['error'] = str(ex)
        else:
            print(traceback.format_exc())
            data['error'] = str(ex)
    data['run_time'] = "%.2f" % (time.time() - start_time)
    return jsonify(data)


@app.route('/call_raw/predict_frontend', methods=['POST'])
def mnist_predict_frontend():
    start_time = time.time()
    data = {}
    if request.method == "POST":
        image = request.form.get("image", None)
        if image is not None:
            try:
                image = image.split("base64,")[1]
                image = BytesIO(base64.b64decode(image))
                image = Image.open(image)
                data = mnist_model.predict_frontend(image)
            except Exception as ex:
                print(traceback.format_exc())
                data['error'] = str(ex)
        else:
            print(traceback.format_exc())
            data['error'] = str(ex)
    data['run_time'] = "%.2f" % (time.time() - start_time)
    return jsonify(data)


@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=3000)
