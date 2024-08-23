from flask import Flask, render_template, request
from astropy.io import fits
import numpy as np
from scipy.stats import skew, kurtosis
import os
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static_files'

@app.route('/', methods=['GET', 'POST'])
def index():
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
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], kepler_id, filename)
    
    # Read FITS file
    hdul = fits.open(filepath)
    data = hdul[1].data
    time = data['TIME']
    flux = data['PDCSAP_FLUX']
    
    # Generate statistics
    statistics = generate_stats(flux)
    
    # Generate visualization
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
    statistics['Skewness'] = skew(flux[~np.isnan(flux)])  # Remove NaN values for skewness calculation
    statistics['Kurtosis'] = kurtosis(flux[~np.isnan(flux)])  # Remove NaN values for kurtosis calculation
    return statistics

def generate_visualization(time, flux):
    # Plot the light curve
    plt.figure(figsize=(8, 6))
    plt.plot(time, flux, color='b')
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.title('Light Curve')
    
    # Convert plot to base64-encoded image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_base64 = base64.b64encode(img.getvalue()).decode()
    img_data = img_base64
    
    return img_data

if __name__ == '__main__':
    app.run(debug=True)
