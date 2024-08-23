from flask import Flask, request, redirect, url_for, flash, render_template
from werkzeug.utils import secure_filename
import os
import zipfile
import subprocess

app = Flask(__name__)
app.secret_key = 'secret_key'  # Required for session management and flash messages
app.config['MAX_CONTENT_LENGTH'] = 12 * 1024 * 1024 * 1024  # 12 GB limit

# Directory where uploaded files will be stored
UPLOAD_FOLDER = 'kepler-2'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'zip'}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_zip(file, destination):
    """Extract a zip file to the specified destination."""
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(destination)

@app.route('/', methods=['GET', 'POST'])
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
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # If the uploaded file is a zip file, extract it
        if filename.lower().endswith('.zip'):
            extract_zip(file_path, app.config['UPLOAD_FOLDER'])
            flash('Zip file successfully uploaded and extracted')
        else:
            flash('File successfully uploaded')

        # Run the process_data.py script as a separate process
        script_path = os.path.join(os.path.dirname(__file__), 'data/generate_input_records_test.py')
        subprocess.Popen(['python', script_path])

        return redirect(url_for('upload_file'))

    return render_template('zip.html')

if __name__ == '__main__':
    app.run(debug=True)
