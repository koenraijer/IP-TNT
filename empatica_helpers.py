import os
import shutil
import re
import pandas as pd
import numpy as np
import resampy

def move_folders_to_root(root_path, current_path=None):
    """
    This function deletes the .DS_Store file, recursively searches through all the folders and subfolders, and moves
    folders with no subfolders to the root directory. Empty folders are deleted.
    
    :param root_path: The root path where the parent folders are located.
    :param current_path: The current path being processed (used for recursion). Defaults to None.
    """
    if current_path is None:
        current_path = root_path

        # Check if the root_path already contains only files and no folders
        items_in_root = [os.path.join(root_path, item) for item in os.listdir(root_path)]
        if all(os.path.isfile(item) for item in items_in_root):
            return
        
    # Remove the .DS_Store file if it exists
    ds_store_path = os.path.join(current_path, '.DS_Store')
    if os.path.exists(ds_store_path):
        os.remove(ds_store_path)

    subfolders = [os.path.join(current_path, item) for item in os.listdir(current_path) if os.path.isdir(os.path.join(current_path, item))]
    
    if not subfolders:
        # Base case: no subfolders
        if current_path != root_path:
            # Move the folder to the root directory
            shutil.move(current_path, get_unique_path(os.path.join(root_path, os.path.basename(current_path))))
    else:
        # Recursively process subfolders
        for subfolder in subfolders:
            move_folders_to_root(root_path, subfolder)

        # Remove empty folders
        if not os.listdir(current_path) and current_path != root_path:
            os.rmdir(current_path)

def get_unique_path(path):
    """
    This function generates a unique path if the given path already exists.
    
    :param path: The path to check for uniqueness.
    :return: A unique path.
    """
    counter = 1
    unique_path = path
    while os.path.exists(unique_path):
        unique_path = f"{path}_{counter}"
        counter += 1
    return unique_path

def clean_folder_name(folder_name):
    # Find all occurrences of '_<number>'
    matches = re.findall(r'_(\d+)', folder_name)
    
    # If there's more than one occurrence, remove all except the first one
    if len(matches) > 1:
        for match in matches[1:]:
            folder_name = folder_name.replace(f'_{match}', '', 1)
    
    return folder_name

def rename_folders(parent_folder):
    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)
        
        # Check if it's a folder
        if os.path.isdir(item_path):
            cleaned_name = clean_folder_name(item)
            cleaned_path = os.path.join(parent_folder, cleaned_name)
            
            # Ensure the cleaned folder path is unique
            cleaned_path = get_unique_path(cleaned_path)
            
            # Rename the folder
            if item_path != cleaned_path:
                os.rename(item_path, cleaned_path)

def load_data_and_combine(folder, common_sr = 4):
    print(folder)
    file_names = ['ACC.csv', 'HR.csv', 'TEMP.csv', 'EDA.csv', 'BVP.csv', ]
    data_frames = []
    starting_times = []
    ending_times = []

    # Read the CSV files and extract starting times, sampling rates, and data
    for file_name in file_names:
        file_path = os.path.join(folder, file_name)
        data = pd.read_csv(file_path, header=None)

        starting_time = data.iloc[0, 0]
        sampling_rate = int(data.iloc[1, 0])
        signal = data.iloc[2:, :].values

        # Check if the signal length is large enough
        if signal.shape[0] >= 2:
            signal_data = resampy.resample(signal, sampling_rate, common_sr, axis=0)
            signal_data = pd.DataFrame(signal_data)

            # Set column names
            if file_name == 'ACC.csv':  # Handle the ACC.csv file with 3 columns
                signal_data.columns = ['acc_x', 'acc_y', 'acc_z']
            else:
                column_name = file_name.lower().replace('.csv', '')
                signal_data.columns = [column_name]

            starting_times.append(starting_time)
            ending_times.append(starting_time + signal_data.shape[0] / common_sr)

            data_frames.append(signal_data)
        else:
            print(f"Skipping {file_name} in {folder} due to insufficient signal length.")


    # Trim dataframes
    trimmed_data_frames, latest_start_time, earliest_end_time = trim_dataframes(data_frames, starting_times, ending_times)

    reset_index_data_frames = [df.reset_index(drop=True) for df in trimmed_data_frames]
    concatenated_df = pd.concat(reset_index_data_frames, axis=1)

    # Add time column starting at latest_start_time and ending at earliest_end_time
    time_unix = np.arange(latest_start_time, earliest_end_time, 1 / common_sr)
    time = pd.to_datetime(time_unix, unit='s', origin='unix')
    # Check number of unique values in time
    concatenated_df['time'] = time

    return concatenated_df

def trim_dataframes(data_frames, starting_times, ending_times, sr=4):
    latest_start_time = max(starting_times)
    earliest_end_time = min(ending_times)

    trimmed_data_frames = []
    for idx, df in enumerate(data_frames):
        start_idx = int((latest_start_time - starting_times[idx]) * sr)
        end_idx = int((earliest_end_time - starting_times[idx]) * sr)
        df_trimmed = df.iloc[start_idx:end_idx, :]
        trimmed_data_frames.append(df_trimmed)

    return trimmed_data_frames, latest_start_time, earliest_end_time



"""
I have different biosignals, each with different sampling sampling rates. I'll give an example for just EDA  (4 Hz) and TEMP (1 Hz):
```
EDA, TEMP
2, 32.1
1.8, NaN
1.5, NaN
1.3, NaN
1.4, 32.3
```

Please finish the following function
"""

"""
If 
"""