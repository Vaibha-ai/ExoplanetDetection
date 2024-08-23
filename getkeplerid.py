import os

def get_all_subfolders(directory):
    all_subfolders = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(directory):
        for subdir in dirs:
            # Construct the full path of the subfolder
            subfolder_path = os.path.join(root, subdir)
            all_subfolders.append(subfolder_path)

    return all_subfolders

# Specify the directory path
directory_path = 'kepler'

# Get the list of all subfolders in the specified directory and its subdirectories
all_subfolders_list = get_all_subfolders(directory_path)

# Print the list of all subfolders
print("List of All Subfolders:")
for subfolder in all_subfolders_list:
    print(subfolder)
