import pickle
import os

def load_data(filename):
    file_path = os.path.join(os.path.dirname(__file__), filename)
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def get_data_filenames():
    train_data_dir = os.path.join(os.path.dirname(__file__), 'train')
    all_files = os.listdir(train_data_dir)
    data_files = [f for f in all_files if f.endswith('.pkl')]

    return data_files
