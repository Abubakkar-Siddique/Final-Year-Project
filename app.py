import numpy as np
import os
from keras.models import model_from_json
import matplotlib.pyplot as plt
import os
import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import BernoulliNB
from keras.models import load_model
import imutils
from skimage.transform import resize
from flask import Flask, request, jsonify, render_template
import pickle

import Wavlet as wavelet

app = Flask(__name__, template_folder="templates", static_folder="static")

model_NA_SVM = pickle.load(open('Normal_Abnormal_SVM_model.sav', 'rb'))
model_FC_SVM = pickle.load(open('Fatty_Cirosis_SVM_model.sav', 'rb'))


@app.route('/', methods=['POST', 'GET'])
def pre():

    if request.method == "POST":
        try:
            inputs = request.files['inpFile']
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', inputs.filename)
            inputs.save(file_path)
        except FileNotFoundError:
            print()

        image = plt.imread(file_path)

        if len(image.shape) >= 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        wavelet.featureExtraction(image)
        features = wavelet.normal_vs_abnormal()
        features = features.reshape(1, -1)
        prediction_NA = model_NA_SVM.predict(features)

        if (prediction_NA == 1):

            features = wavelet.Fatty_vs_cirrhotic()
            features = features.reshape(1, -1)

            prediction_FC = model_FC_SVM.predict(features)
            if prediction_FC == 0:
                return render_template("index.html", Predicted_result="Fatty")
            return render_template("index.html", Predicted_result="Cirrhotic")

        return render_template("index.html", Predicted_result="Normal")
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
