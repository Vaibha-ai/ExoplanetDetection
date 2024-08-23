from flask import Flask, request, render_template, Response
import os
import subprocess

app = Flask(__name__)

UPLOAD_FOLDER = 'astronet/uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_folder(filename):
    # Allow any folder to be uploaded
    return True

def generate_output(proc):
    for line in iter(proc.stdout.readline, b''):
        yield line

@app.route('/', methods=['GET', 'POST'])
def upload_folder():
    if request.method == 'POST':
        # check if the post request has the folder part
        if 'file' not in request.files:
            # No files uploaded, start training with default values
            proc = subprocess.Popen(['python', 'train_modified.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            return Response(generate_output(proc), mimetype='text/plain')
        files = request.files.getlist('file')
        for file in files:
            if file.filename == '':
                # No files uploaded, start training with default values
                proc = subprocess.Popen(['python', 'train_modified.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                return Response(generate_output(proc), mimetype='text/plain')
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
        # Call the training script with uploaded folder path
        proc = subprocess.Popen(['python', 'train_modified.py', '--uploaded_folder', UPLOAD_FOLDER], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return Response(generate_output(proc), mimetype='text/plain')
    return render_template('train_page_upload.html')

if __name__ == '__main__':
    app.run(debug=True)
