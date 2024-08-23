from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import subprocess

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            kepler_id = request.form['kepler_id']
            period = request.form['period']
            t0 = request.form['t0']
            duration = request.form['duration']
            
            # Get uploaded folder
            folder = request.files['folder']
            folder_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(folder.filename))
            folder.save(folder_path)
            
            # Extract directory name from the uploaded folder path
            kepler_data_dir = os.path.basename(folder_path)
            
            # Call the script for generating predictions
            result = subprocess.run([
                "python", "predict.py",
                "--model", "AstroCNNModel",
                "--config_name", "local_global",
                "--model_dir", "astronet/model",
                "--kepler_data_dir", kepler_data_dir,
                "--kepler_id", kepler_id,
                "--period", period,
                "--t0", t0,
                "--duration", duration
            ], capture_output=True)
            
            if result.returncode == 0:
                prediction = result.stdout.decode('utf-8').strip()
                # Process the prediction as needed
                return render_template('result.html', prediction=prediction)
            else:
                error_message = result.stderr.decode('utf-8').strip()
                return render_template('error.html', error=error_message)
        except Exception as e:
            print("An error occurred:", e)  # Log the error message
            return jsonify({'error': 'An error occurred while processing your request.'}), 500
    return render_template('test.html')

if __name__ == '__main__':
    app.run(debug=True)
