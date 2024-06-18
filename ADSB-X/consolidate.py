import os
import pandas as pd
import numpy as np
import json
import pickle
import shutil
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

##############################
# Defining Helpers Functions #
##############################

def get_leaf_folders(root):
    """
    Retrieves the longest paths in the root directory.
    """
    paths = []
    for root, dirs, files in os.walk(root):
        for name in dirs:
            paths.append(os.path.join(root, name))

    paths = [path for path in paths if 'consolidated' not in path]
    max_length = max(len(path) for path in paths)
    longest_paths = [path for path in paths if len(path) == max_length]

    return longest_paths

def extract_timestamp(filepath):
    # Split the filepath into parts
    parts = filepath.split('/')
    
    # Extract the date and time parts
    date_parts = parts[-4:-1]  # ['2020', '03', '25']
    time_part = parts[-1].split('.')[0]  # '013200Z'
    
    # Combine the date and time parts into a timestamp
    timestamp_str = '-'.join(date_parts) + ' ' + time_part[:-1]
    
    # Convert the timestamp string to a pandas Timestamp object
    timestamp = pd.to_datetime(timestamp_str, format='%Y-%m-%d %H%M%S')
    
    return timestamp

def get_file_names(path):
    """ Retrieves filenames in a directory and its subdirectories."""
    return [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]


def save_file(filepath, data:pd.DataFrame) -> None:
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)

def load_file(filepath) -> pd.DataFrame:
    with open(filepath, 'rb') as file:
        data = pickle.load(file)
    return data

def clean_root(root_dir):
    """
    Cleans the raw data directory by removing all directories except the consolidated directory.
    """
    dirs = os.listdir(root_dir)
    for dir in dirs:
        if dir != 'consolidated':
            shutil.rmtree(os.path.join(root_dir, dir))

def create_df(data):
    """
    Creates a dataframe from the json data.
    In this context, the json data corresponds to one timestamp.
    Filters flights that are not of type 'A3*' or 'B7*'.
    """
    df = pd.json_normalize(data['aircraft'])
    df['timestamp'] = pd.to_datetime(data['now'], unit='s')
    df.set_index('timestamp', inplace=True)
    df = df.replace({None: np.nan})
    df = df.astype({
        'flight': 'object',
        'type': 'object',
        'hex': 'object',
        'r': 'object',
        't': 'object',
        'gs': 'float64',
        'track': 'float64',
        'baro_rate': 'float64',
        'alt_geom': 'float64'
    })
    df = df[df['t'].str.startswith('A3') | df['t'].str.startswith('B7')]
    def convert_alt_baro(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(-1)

    # Apply the conversion function
    df['alt_baro'] = df['alt_baro'].apply(convert_alt_baro)
    df = df.sort_index()
    return df

def read_json_file(filepath):
    with open(filepath, 'r') as file:
        data = json.load(file)
    return create_df(data)

def logic(path) -> None :
    """
    Reads the json files from the root directory and creates a dataframe from them.
    Consolidates data into monthly files.

    Pickle files: pd.Dataframes

    """
    longest_paths = get_leaf_folders(path)
    longest_paths = [path for path in longest_paths if '2020' not in path and '2021' not in path]
    for path in tqdm(longest_paths, desc='Processing ...'):
        file_names = sorted(get_file_names(path))
        dfs = []
        timestamp = extract_timestamp(os.path.join(path, file_names[0]))
        year = timestamp.year
        month = timestamp.month
        output_fname = f"data/consolidated/{year:04d}/{month:02d}.pkl"
        for file in file_names:
            dfs.append(read_json_file(os.path.join(path, file)))
        # with ThreadPoolExecutor(max_workers=10) as executor:
        #         file_paths = [os.path.join(path, file) for file in file_names]
        #         results = list(executor.map(read_json_file, file_paths))
        df = pd.concat(dfs)
        # Saving df to hdf file
        timestamp = df.iloc[0].name
        year = str(timestamp.year)
        month = str(timestamp.month).zfill(2)
        os.makedirs(f"data/consolidated/{year}", exist_ok=True)

        with open(output_fname, 'wb') as f:
            pickle.dump(df, f)

if __name__ == '__main__':
    logic('data')


