import pickle

# Specify the path to the pickle file
pickle_file_path = 'astronet/model/test.pkl'

# Load the model from the pickle file
with open(pickle_file_path, 'rb') as f:
    loaded_model = pickle.load(f)

# Print relevant details or attributes of the loaded model
print("Loaded model details:")
print("Model type:", type(loaded_model))
# Print other relevant details as needed
