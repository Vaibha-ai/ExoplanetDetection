from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import pickle
import numpy as np
import pandas as pd
import json
from flask import Flask, request, redirect, url_for, flash, render_template, Response
from werkzeug.utils import secure_filename
import zipfile
import subprocess
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
from collections import defaultdict
import base64
import sys
from scipy.stats import skew, kurtosis
import pickle
import json
import plotly.graph_objects as go
from tensorflow.compat.v1.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024 * 1024  # 12 GB limit
app.config['UPLOAD_FOLDER'] = 'static_files'
app.config['UPLOAD_FOLDER_REC'] = 'static_files_rec'
MODEL_NAME = 'AstroCNNModel'
#MODEL_DIR = 'astronet/model_modified'
estimator = None
FOLDER = 'astronet/kepler-2'
app.config['FOLDER'] = FOLDER
TRAINF = 'astronet/testing_tfrecords'
app.config['TFRECORDS'] = TRAINF
directory_path = 'static_files_rec'
folder_names = [folder for folder in next(os.walk(directory_path))[1] if folder.startswith('kpl')]



# Ensure upload folder exists
os.makedirs(FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'zip'}

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
@app.route('/pg')
def pg():
    return render_template('page1.html')
@app.route('/rec')
def rec():
    return render_template('rec_train_test.html')

@app.route('/rec_upload', methods=['GET', 'POST'])
def rec_upload():
    form = UploadFileForm()
    if form.validate_on_submit():
        files = form.files.data  # Grab the files
        for file in files:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER_REC'], filename))  # Save each file inside the subfolder
            
        try:
            subprocess.run(['python', 'D:/testing_rec/input_to_folders.py'], check=True)
            subprocess.run(['python', 'D:/testing_rec/window_slicing.py'], check=True)
            subprocess.run(['python', 'D:/testing_rec/delete_empty_files.py'], check=True)
            subprocess.run(['python', 'D:/testing_rec/recurrence_values.py'], check=True)
            subprocess.run(['python', 'D:/exoplanet_ml_master/exoplanet_ml/static_files_rec/testing_csv_generation.py'], check=True)
            subprocess.run(['python', 'D:/exoplanet_ml_master/exoplanet_ml/static_files_rec/time_collect.py'], check=True)
        except subprocess.CalledProcessError as e:
            return f'Error running script: {e}'
    tf.keras.backend.clear_session()
    return render_template('rec_upload.html', form=form)


@app.route('/rec_measure_plot', methods=['GET', 'POST'])
def rec_plot():
    global folder_names
    current_index = int(request.form.get('current_index', 0))
    
    if request.method == 'POST':
        action = request.form['action']
        
        if action == 'Previous':
            current_index -= 1
        elif action == 'Next':
            current_index += 1

        if current_index < 0:
            current_index = 0
        elif current_index >= len(folder_names):
            current_index = len(folder_names) - 1

    current_folder = folder_names[current_index]

    data = np.genfromtxt(f'{directory_path}/{current_folder}/final_results.txt')
    time_data = np.genfromtxt(f'{directory_path}/{current_folder}/time.txt')

    column_names = ['Recurrence Rate', 'Determinism', 'Average L', 'Maximum Length']
    num_columns = data.shape[1]

    fig1 = go.Figure()
    for i in range(num_columns):
        fig1.add_trace(go.Scatter(x=time_data, y=data[:, i], mode='lines+markers', name=column_names[i]))
    fig1.update_layout(title=f'Recurrence Measures', xaxis_title='Time', yaxis_title='Values')

    hdul = fits.open(f'{directory_path}/{current_folder}.fits')
    data = hdul[1].data
    time = data['TIME']
    flux = data['PDCSAP_FLUX']

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=time, y=flux, mode='markers', name='Light Curve'))
    fig2.update_layout(title='Light Curve', xaxis_title='Time', yaxis_title='Flux')
    hdul.close()

    return render_template('rec_measures.html', 
                           fig1=fig1.to_html(full_html=False), 
                           fig2=fig2.to_html(full_html=False), 
                           current_index=current_index, 
                           folder_name=current_folder,
                           folder_names=folder_names)


label_mapping = {
    0: 0,
    60: 1,
    65: 2,
    70: 3,
    75: 4,
    80: 5,
    85: 6,
    90: 7
}

def load_tf1_model(model_path):
    tf.keras.backend.clear_session()
    global graph
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        model = load_model(model_path)
    return model

# Ensure to use the correct path to your saved model
model_path = "D:/exoplanet_ml_master/exoplanet_ml/trained_model_tf1x.h5"
print("Checking if model path exists:", model_path)
print("Model path exists:", os.path.exists(model_path))

if os.path.exists(model_path):
    print("Attempting to load model...")
    try:
        model = load_tf1_model(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print("Error loading model:", e)
else:
    print("Model path is not accessible. Check permissions and path.")

@app.route('/rec_prediction', methods=['GET'])
def rec_predict():
    folder_path = 'static_files_rec'
    csv_file = os.path.join(folder_path, 'final_output.csv')
    time_csv_file = os.path.join(folder_path, 'final_time.csv')
    
    print("CSV file path:", csv_file)
    print("Time CSV file path:", time_csv_file)
    
    # Read the CSV files
    df = pd.read_csv(csv_file)
    df_time = pd.read_csv(time_csv_file)
    
    # Drop unnecessary columns
    df.drop(['kepid_licurve', 'slice_number'], axis=1, inplace=True)
    
    # Reshape the data
    df_reshaped = df.values.reshape(df.shape[0], df.shape[1], 1)
    
    # Load the model inside the prediction function
    model_path = "D:/exoplanet_ml_master/exoplanet_ml/trained_model_tf1x.h5"
    model = load_tf1_model(model_path)
    
    # Make prediction
    with graph.as_default():
        predictions = model.predict(df_reshaped)
    
    # Get predicted classes
    y_pred_classes = np.argmax(predictions, axis=1)
    
    # Create a copy of df_time DataFrame
    df_time_with_labels = df_time.copy()
    
    # Add predicted classes to df_time_with_labels DataFrame
    df_time_with_labels['labels'] = y_pred_classes
    
    # Save the updated DataFrame to a new CSV file
    final_time_with_labels_csv_file = os.path.join(folder_path, 'final_time_with_labels.csv')
    df_time_with_labels.to_csv(final_time_with_labels_csv_file, index=False)
    
    # Compute the percentage of each label's presence
    label_counts = df_time_with_labels['labels'].value_counts(normalize=True) * 100
    
    # Convert label counts to a DataFrame for easier rendering
    label_counts_df = pd.DataFrame({
        'Label': label_counts.index,
        'Percentage': label_counts.values
    })
    
    # Render prediction result template
    return render_template('rec_predict.html', df_time_with_labels=df_time_with_labels, label_counts=label_counts_df)

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
    return render_template('test.html', form=form)

@app.route('/algo',methods=['GET','POST'])
def algo():
    return render_template('algo.html')

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
    for root, dirs, filenames in os.walk(app.config['UPLOAD_FOLDER']):
        for filename in filenames:
            files.append(filename)
    return files

def generate_stats_and_visualization(filename):
    kepler_id = filename.split('-')[0].lstrip('kplr')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'],kepler_id[0:4],kepler_id, filename)
    print(kepler_id,filepath)
    current_dir = os.getcwd()
    print(current_dir)
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
    
    # Get the model type from the form data
    model_type = request.args.get('model_type')
    
    kepler_data_dir = request.args.get('kepler_data_dir')
    kepler_id = int(request.args.get('kepler_id'))
    period = float(request.args.get('period'))
    t0 = float(request.args.get('t0'))
    duration = float(request.args.get('duration'))
    
    # Set the MODEL_DIR based on the selected model type
    MODEL_DIR = model_type
    
    # Reload the estimator if MODEL_DIR has changed
    if estimator is None or estimator.model_dir != MODEL_DIR:
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
        
def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_zip(file, destination):
    """Extract a zip file to the specified destination."""
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(destination)

def allowed_folder(filename):
    # Allow any folder to be uploaded
    return True        
        
def generate_output(proc):
    for line in iter(proc.stdout.readline, b''):
        yield line

@app.route('/train', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
       
        file = request.files['file']

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
       
        # Check if the file has an allowed extension
        if not allowed_file(file.filename):
            flash('File type not allowed')
            return redirect(request.url)

        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['FOLDER'], filename)
        file.save(file_path)

        # If the uploaded file is a zip file, extract it
        if filename.lower().endswith('.zip'):
            extract_zip(file_path, app.config['FOLDER'])
            flash('Zip file successfully uploaded and extracted')
        else:
            flash('File successfully uploaded')

        # Run the process_data.py script as a separate process
        script_path = os.path.join(os.path.dirname(__file__), 'astronet/data/generate_input_records_test.py')
        subprocess.Popen(['python', script_path]).wait()  # Wait for the first subprocess to finish

        # Now, run the train_modified.py script
        train_script_path = os.path.join(os.path.dirname(__file__), 'train_modified.py')
        proc = subprocess.Popen(['python', train_script_path, '--uploaded_folder', TRAINF], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return Response(generate_output(proc), mimetype='text/plain')

    return render_template('zip.html')

    
if __name__ == '__main__':
    app.run(debug=True)