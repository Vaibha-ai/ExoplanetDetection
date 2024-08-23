from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pickle
import numpy as np
import json

import tensorflow as tf
from astronet.data import preprocess
from flask import Flask, request, jsonify, render_template

class NumpyFloat32Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.dtype == np.float32:
            return obj.astype(float).tolist()  # Convert float32 numpy array to Python list
        return json.JSONEncoder.default(self, obj)

app = Flask(__name__)

# Define model and model directory
MODEL_NAME = 'AstroCNNModel'
MODEL_DIR = 'astronet/model'

estimator = None

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['GET'])
def predict():
    global estimator

    
    # Get input parameters from the request
    kepler_data_dir = request.args.get('kepler_data_dir')
    kepler_id = int(request.args.get('kepler_id'))
    period = float(request.args.get('period'))
    t0 = float(request.args.get('t0'))
    duration = float(request.args.get('duration'))

    # Load the estimator if not loaded
    if estimator is None:
        load_estimator_from_pickle(MODEL_DIR)

    # Read input features
    features = _process_tce(kepler_data_dir, kepler_id, period, t0, duration)

    # Create an input function
    def input_fn():
        return tf.data.Dataset.from_tensors({"time_series_features": features})

    # Generate the predictions
    predictions_list = []
    for predictions in estimator.predict(input_fn):
        predictions_list.append(float(predictions[0]))

    # Return prediction value
    return jsonify(predictions_list)

def _process_tce(kepler_data_dir, kepler_id, period, t0, duration):
    """Reads and processes the input features of a Threshold Crossing Event."""
    # Read and process the light curve.
    all_time, all_flux = preprocess.read_light_curve(kepler_id, kepler_data_dir)
    time, flux = preprocess.process_light_curve(all_time, all_flux)
    time, flux = preprocess.phase_fold_and_sort_light_curve(time, flux, period, t0)

    # Generate the local and global views.
    features = {}

    global_view = preprocess.global_view(time, flux, period).astype(np.float32)
    features["global_view"] = np.expand_dims(global_view, 0)

    local_view = preprocess.local_view(time, flux, period, duration).astype(np.float32)
    features["local_view"] = np.expand_dims(local_view, 0)

    return features

def load_estimator_from_pickle(model_dir):
    global estimator
    # Load the model from the pickle file
    with open(model_dir + '/test.pkl', 'rb') as f:
        estimator = pickle.load(f)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
