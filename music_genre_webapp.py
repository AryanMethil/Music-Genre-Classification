from flask import Flask, request
import numpy as np
import tensorflow as tf
import pandas as pd
import flasgger
from flasgger import Swagger
from math import *
import librosa
import statistics
import json

app = Flask(__name__)
Swagger(app)

model = tf.keras.models.load_model("C:\\Users\\ASUS\\PycharmProjects\\College\\my_music_model_1.h5")

with open('C:\\Users\\ASUS\\PycharmProjects\\College\\dataset.json','r') as f:
  data = json.load(f)
  label_to_genre = data['mapping']


#Initializing mfcc variables

n_mfcc=13                                                                       #number of mfcc coefficients
n_fft=2048                                                                      #number of samples per fft
hop_length=512                                                                  #frame 2 will begin from 512th sample and so on
num_segments=10                                                                 #number of segments we want to divide the track into
sample_rate=22050                                                               #sampling rate
duration=30                                                                     #each track in every genre is 30 seconds long
samples_per_track=sample_rate*duration
samples_per_segment=samples_per_track//num_segments
expected_num_mfcc_vectors_per_segment= ceil(samples_per_segment/hop_length)


@app.route('/')
def welcome():
    return "Music Genre Classification WebApp"


@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    """Let's classify the music genres
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true

    responses:
        200:
            description: The output values

    """
    file = request.files.get("file")
    signal, sr = librosa.load(file, sr=sample_rate)

    mfcc_arrays = []
    for s in range(num_segments):
        start_segment = samples_per_segment * s
        end_segment = start_segment + samples_per_segment

        mfcc_array = librosa.feature.mfcc(signal[start_segment:end_segment],
                                          sr=sr,
                                          n_fft=n_fft,
                                          n_mfcc=n_mfcc,
                                          hop_length=hop_length)
        mfcc_array = mfcc_array.T
        mfcc_arrays.append(mfcc_array)

    prediction = model.predict(np.array(mfcc_arrays))
    prediction_list =[np.argmax(arr) for arr in prediction]

    final_label = statistics.mode(prediction_list)
    return label_to_genre[final_label]


if __name__ == '__main__':
    app.run(debug=True)
