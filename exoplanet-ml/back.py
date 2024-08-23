from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import pickle
import tensorflow as tf
import numpy as np
from astronet.data import preprocess  # if preprocess module is part of the astronet package

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static'
estimator = None  # Initialize estimator as global variable
MODEL_NAME = 'AstroCNNModel'
MODEL_DIR = 'astronet/model'

class UploadFileForm(FlaskForm):
    kepler_id = IntegerField("Kepler ID", validators=[InputRequired()])
    files = MultipleFileField("Files", validators=[InputRequired()])
    submit = SubmitField("Upload Files")

@app.route('/', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get form data
            kepler_id = request.form['kepler_id']
            files = request.files.getlist('folder')

            # Create folder if not exists
            folder_path = os.path.join(app.config['UPLOAD_FOLDER'], '{:09d}'.format(int(kepler_id)))
            os.makedirs(folder_path, exist_ok=True)

            # Save each file into the folder
            for file in files:
                filename = secure_filename(file.filename)
                file.save(os.path.join(folder_path, filename))

            return "Files have been uploaded."
        except Exception as e:
            app.logger.error("An error occurred while processing file upload: %s", str(e))
            return jsonify({'error': 'An error occurred while processing your request.'}), 500

    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    global estimator

    try:
        # Get input parameters from the request
        kepler_data_dir = request.form['kepler_data_dir']
        kepler_id = int(request.form['kepler_id'])
        period = float(request.form['period'])
        t0 = float(request.form['t0'])
        duration = float(request.form['duration'])

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
    except Exception as e:
        app.logger.error("An error occurred while processing prediction: %s", str(e))
        return jsonify({'error': 'An error occurred while processing your request.'}), 500

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

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        app.logger.error("An unhandled exception occurred: %s", str(e))
