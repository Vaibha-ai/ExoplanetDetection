import subprocess
from lightkurve import search_lightcurvefile
import os

def download_kepler_lightcurves(kepler_ids, download_dir='.'):
    """
    Download Kepler light curve files for a list of Kepler IDs.

    Parameters:
    - kepler_ids (list): List of Kepler IDs of the target stars.
    - download_dir (str): Directory to save the downloaded files.

    Returns:
    - List of downloaded file paths.
    """
    downloaded_files = []

    for kepler_id in kepler_ids:
        # Pad Kepler ID with zeros to make it 9 digits long
        kepler_id_padded = str(kepler_id).zfill(9)
        #print(f"Padded kepler_id{kepler_id_padded}")
        # Extract the first 4 digits
        subdir = kepler_id_padded[:4]

        # Create the directory structure
        kepler_dir = os.path.join(download_dir, subdir)
        os.makedirs(kepler_dir, exist_ok=True)  # Create the folder if it doesn't exist
        
        # Create a new folder with the entire Kepler ID as the folder name
        kepler_subdir = os.path.join(kepler_dir, kepler_id_padded)
        os.makedirs(kepler_subdir, exist_ok=True)  # Create the folder if it doesn't exist
        
        # Search for available light curve files for the given Kepler ID
        search_str = f'KIC {kepler_id}'

        # Construct wget command
        wget_command = [
            'wget', '-q', '-nH', '--cut-dirs=6', '-r', '-l0', '-c', '-N', '-np',
            '-erobots=off', '-R', 'index*', '-A', '_llc.fits', '-P', kepler_subdir,
            f'http://archive.stsci.edu/pub/kepler/lightcurves/{subdir}/{kepler_id_padded}/'
        ]
        
        # Execute wget command
        subprocess.run(wget_command)
        
        # Uncomment the following line if you want to collect downloaded file paths
        # downloaded_files.extend([os.path.join(kepler_subdir, filename) for filename in os.listdir(kepler_subdir)])

    return downloaded_files

# Example usage:
kepler_ids = [2306740, 2307197, 2554867, 2693092, 2440757, 2569494, 2707985, 2708156, 2581316, 2718630, 2159852, 2711123, 3547091, 3644523, 3540153, 3328027, 3345675, 3456780, 3543070, 3553413, 3544595, 3747641, 4679769, 4774194, 4842166, 4816098, 4819564, 4820642, 4851457, 4636578, 4847411, 4456940, 4578644, 4861784, 4371947, 3965201, 4271063, 4252322, 3865358, 5566778, 5481148, 5649837, 5444549, 5482030, 5601258, 5470960, 5552485, 5631630, 5385187, 5395490, 5617110, 5215508, 5374838, 5264764, 5358323, 5211470, 5039687, 5281113, 5353938, 6301749, 6364067, 6305192, 6350476, 6382943, 6435936, 5865766, 5796675, 5817243, 5728283, 5717140, 5872972, 5894549, 7100673, 7118545, 7137952, 7018210, 7031656, 7035274, 7281699, 7107802, 7117270, 7215603, 6962018, 6778274, 6470149, 6632383, 6776957, 7872212, 7919763, 8081566, 8038388, 8172936, 7458762, 7505674, 7362534, 7514582, 7463737, 7630232, 7609674, 7659782, 8505554, 8488876, 8628973, 8553462, 8610483, 8509442, 8494783, 8542993, 8559644, 8689793, 8314392, 8288719, 8255272, 8329728, 8378634, 8380242, 8443132, 8461529, 8398221, 8349399, 9479537, 9288237, 9343862, 9514372, 9487546, 9345819, 9426817, 9573539, 9474483, 8842025, 8981233, 8951965, 9033798, 9093349, 9146018, 9002278, 8891395, 9152999, 8934103, 10268835, 10271806, 10148787, 10154966, 10263705, 10484817, 10353968, 10015937, 9839081, 9964670, 9773270, 9899141, 9713213, 9718066, 9850409, 9655769, 9654468, 9655909, 11457191, 11520793, 11298220, 11357664, 11080711, 11403044, 11515221, 11564882, 10880895, 10990961, 10661721, 10661917, 10977671, 10583181, 10521496, 10735519, 11025754, 10922610, 11029626, 10879833, 11811382, 11854636, 11770258, 11973921, 11818800, 12066447, 12110942, 12316431, 12505503, 12069414]  # List of Kepler IDs
download_directory = 'astronet/kepler'
downloaded_files = download_kepler_lightcurves(kepler_ids, download_directory)

print(f"Downloaded {len(downloaded_files)} file(s): {downloaded_files}")
