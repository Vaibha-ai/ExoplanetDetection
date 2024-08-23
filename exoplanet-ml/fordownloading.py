"""from lightkurve import search_lightcurvefile

def download_kepler_lightcurve(kepler_id, download_dir='.', quarter=None):

    # Search for available light curve files for the given Kepler ID
    if quarter is not None:
        search_str = f'KIC {kepler_id} q{quarter}'
    else:
        search_str = f'KIC {kepler_id}'
    
    lcfs = search_lightcurvefile(search_str).download_all(download_dir=download_dir)

    # Return the list of downloaded file paths
    return [lcf.filename for lcf in lcfs]

# Example usage:
kepler_id = 5894549
download_directory = 'astronet/kepler1'
downloaded_files = download_kepler_lightcurve(kepler_id, download_directory)

print(f"Downloaded {len(downloaded_files)} file(s): {downloaded_files}")
"""
from lightkurve import search_lightcurvefile

def download_kepler_lightcurve(kepler_id, download_dir='.'):
    """
    Download all available Kepler light curve files for a given Kepler ID.

    Parameters:
    - kepler_id (int): Kepler ID of the target star.
    - download_dir (str): Directory to save the downloaded files.

    Returns:
    - List of downloaded file paths.
    """
    search_str = f'KIC {kepler_id}'
    lcfs = search_lightcurvefile(search_str).download_all(download_dir=download_dir)

    # Return the list of downloaded file paths
    return [lcf.filename for lcf in lcfs]

# Example usage:
kepler_id = 11442793
download_directory = 'astronet/kepler1'
downloaded_files = download_kepler_lightcurve(kepler_id, download_directory)

print(f"Downloaded {len(downloaded_files)} file(s): {downloaded_files}")



