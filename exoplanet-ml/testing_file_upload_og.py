from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pickle
import numpy as np
import json

import tensorflow as tf
from astronet.data import preprocess
from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_wtf import FlaskForm
from wtforms import MultipleFileField, IntegerField, SubmitField
from wtforms.validators import InputRequired
from werkzeug.utils import secure_filename
import os
from astropy.io import fits
import matplotlib.pyplot as plt
import io
import base64
import sys
from scipy.stats import skew, kurtosis
import pickle
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static_files'
MODEL_NAME = 'AstroCNNModel'
MODEL_DIR = 'astronet/model'
estimator = None

class NumpyFloat32Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.dtype == np.float32:
            return obj.astype(float).tolist()  # Convert float32 numpy array to Python list
        return json.JSONEncoder.default(self, obj)

class UploadFileForm(FlaskForm):
    kepler_id = IntegerField("Kepler ID", validators=[InputRequired()])
    files = MultipleFileField("Files", validators=[InputRequired()])
    submit = SubmitField("Upload Files")

@app.route('/')
def index():
    return render_template('index.html')

    
@app.route('/home', methods=['GET', 'POST'])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        kepler_id = '{:09d}'.format(form.kepler_id.data)  # Pad with zeros to make it 9 digits
        # Take the first 4 digits
        folder_prefix = kepler_id[:4]
        
        # Create folder with the first 4 digits of Kepler ID
        folder_path = os.path.join(app.config['UPLOAD_FOLDER'], folder_prefix)
        os.makedirs(folder_path, exist_ok=True)  # Create folder if not exists
        
        # Create subfolder with the entire Kepler ID inside
        subfolder_path = os.path.join(folder_path, kepler_id)
        os.makedirs(subfolder_path, exist_ok=True)
        
        files = form.files.data  # Grab the files
        for file in files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(subfolder_path, filename))  # Save each file inside the subfolder
        return redirect(url_for('visualize'))
    return render_template('test.html', form=form)


@app.route('/visualize', methods=['GET', 'POST'])

def visualize():
    files = get_files()
    if request.method == 'POST':
        selected_file = request.form['file']
        statistics, img_data = generate_stats_and_visualization(selected_file)
        return render_template('displaying_fits.html', files=files, result=statistics, img_data=img_data)
    return render_template('displaying_fits.html', files=files)

def get_files():
    files = []
    root_folder = app.config['UPLOAD_FOLDER']
    for root, dirs, filenames in os.walk(root_folder):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def generate_stats_and_visualization(filename):
    #filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    filepath = filename
    #print("filename -->,",filename)
    #print("filepath --->",filepath)
    hdul = fits.open(filepath)
    data = hdul[1].data
    time = data['TIME']
    flux = data['PDCSAP_FLUX']
    
    statistics = generate_stats(flux)
    img_data = generate_visualization(time, flux)
    
    return statistics, img_data

def generate_stats(flux):
    statistics = {}
    statistics['Mean Flux'] = np.nanmean(flux)
    statistics['Median Flux'] = np.nanmedian(flux)
    statistics['Standard Deviation'] = np.nanstd(flux)
    statistics['Minimum Flux'] = np.nanmin(flux)
    statistics['Maximum Flux'] = np.nanmax(flux)
    statistics['Range'] = np.nanmax(flux) - np.nanmin(flux)
    percentiles = [25, 50, 75]
    for percentile in percentiles:
        statistics[f'{percentile}th Percentile'] = np.nanpercentile(flux, percentile)
    statistics['Skewness'] = skew(flux[~np.isnan(flux)])  
    statistics['Kurtosis'] = kurtosis(flux[~np.isnan(flux)])  
    return statistics

def generate_visualization(time, flux):
    plt.figure(figsize=(8, 6))
    plt.plot(time, flux, color='b')
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.title('Light Curve')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    img_data = img_base64
    
    return img_data


@app.route('/form')
def form():
     return render_template('form.html')
    
@app.route('/predict', methods=['GET'])
def predict():
        global estimator
        
        kepler_data_dir = request.args.get('kepler_data_dir')
        kepler_id = int(request.args.get('kepler_id'))
        period = float(request.args.get('period'))
        t0 = float(request.args.get('t0'))
        duration = float(request.args.get('duration'))
        
        if estimator is None:
            load_estimator_from_pickle(MODEL_DIR)
        
        features = _process_tce(kepler_data_dir, kepler_id, period, t0, duration)

        def input_fn():
            return tf.data.Dataset.from_tensors({"time_series_features": features})

        predictions_list = []
        for predictions in estimator.predict(input_fn):
            predictions_list.append(float(predictions[0]))

        return jsonify(predictions_list)
        
       



def _process_tce(kepler_data_dir, kepler_id, period, t0, duration):
    all_time, all_flux = preprocess.read_light_curve(kepler_id, kepler_data_dir)
    time, flux = preprocess.process_light_curve(all_time, all_flux)
    time, flux = preprocess.phase_fold_and_sort_light_curve(time, flux, period, t0)

    features = {}
    global_view = preprocess.global_view(time, flux, period).astype(np.float32)
    features["global_view"] = np.expand_dims(global_view, 0)
    local_view = preprocess.local_view(time, flux, period, duration).astype(np.float32)
    features["local_view"] = np.expand_dims(local_view, 0)

    return features

def load_estimator_from_pickle(model_dir):
    global estimator
    with open(model_dir + '/test.pkl', 'rb') as f:
        estimator = pickle.load(f)

if __name__ == '__main__':
    app.run(debug=True)
