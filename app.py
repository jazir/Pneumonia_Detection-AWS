
"""
Predict whether the person has Pneumonia or not from chest X-ray images
"""

import os
import numpy as np
import shutil

# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # test

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
SAVED_MODEL = 'vgg16_model.h5'

# Load your trained model
model = load_model(SAVED_MODEL)


def model_predict(main_folder, loaded_model):
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_fig = test_datagen.flow_from_directory(f'{main_folder}', target_size=(224, 224),
                                                batch_size=1, class_mode=None, shuffle=False)
    # flow_from_directory expects the test image to be in a sub-folder of the folder we are passing

    pred = loaded_model.predict(test_fig)
    pred = np.argmax(pred, axis=1)
    if pred == 0:
        pred = "Normal"
    else:
        pred = "Pneumonia"

    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        save_loc = os.path.join(basepath, 'uploads', 'test_figs')
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        else:  # if that folder already exits, clear its contents
            shutil.rmtree(save_loc)
            os.mkdir(save_loc)
        file_path = os.path.join(save_loc, secure_filename(f.filename))

        f.save(file_path)

        # Make prediction
        prediction = model_predict(basepath, model)
        result = prediction
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
