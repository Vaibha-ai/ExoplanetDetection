from flask import Flask, render_template, request
import plotly.graph_objects as go
import numpy as np
from astropy.io import fits
import os

app = Flask(__name__)

# Define the directory path where you want to search for folders
directory_path = 'static_files_rec'
folder_names = [folder for folder in next(os.walk(directory_path))[1] if folder.startswith('kpl')]

@app.route('/', methods=['GET', 'POST'])
def index():
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
    fig1.update_layout(title='Recurrence Measures', xaxis_title='Time', yaxis_title='Values')  # Add folder name as title

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

if __name__ == '__main__':
    app.run(debug=True)
