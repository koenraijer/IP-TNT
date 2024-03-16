import os
import shutil
import re
import pandas as pd
import numpy as np
import neurokit2 as nk
import resampy 

def move_folders_to_root(root_path, current_path=None):
    """
    This function deletes the .DS_Store file, recursively searches through all the folders and subfolders, and moves
    folders with no subfolders to the root directory. Empty folders are deleted.
    
    Args:
        root_path (str): The root path where the parent folders are located.
        current_path (str, optional): The current path being processed (used for recursion). Defaults to None.
    
    Returns:
        None
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
    """
    Cleans the folder name by removing the trailing '_<number>' or '-<number>'.
    If no trailing number is found, the function returns False.

    Args:
    folder_name (str): The name of the folder to be cleaned.

    Returns:
    str or bool: The cleaned folder name, or False if no trailing number is found.
    """
    # Remove the trailing '_<number>' or '-<number>'
    cleaned_name = re.sub(r'[_-](\d+)$', '', folder_name)
    
    # Check if a trailing number was found and removed
    if cleaned_name == folder_name:
        return False
    
    return cleaned_name

def rename_folders(parent_folder):
    """
    Renames all folders within the specified parent folder.

    Args:
        parent_folder (str): The path to the parent folder.

    Returns:
        None
    """
    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)

        # Check if it's a folder
        if os.path.isdir(item_path):
            original_name = item

            cleaned_name = original_name
            previous_cleaned_name = original_name

            while cleaned_name:
                # Clean folder name
                cleaned_name = clean_folder_name(cleaned_name)

                # If there is nothing to clean, break
                if cleaned_name is False:
                    cleaned_name = previous_cleaned_name
                    break

                # Find clean folder name path
                cleaned_path = os.path.join(parent_folder, cleaned_name)

                # If cleaned_path is not unique, set cleaned_name to previous_cleaned_name and break
                if os.path.exists(cleaned_path):
                    cleaned_name = previous_cleaned_name
                    break
                else:
                    previous_cleaned_name = cleaned_name

            cleaned_path = os.path.join(parent_folder, cleaned_name)
            # Rename the folder
            print(f"{original_name} -> {cleaned_name}")
            os.rename(item_path, cleaned_path)

def load_data_and_combine(folder, desired_sampling_rate=64, verbose=False):
    """
    This function loads data from multiple CSV files in a given folder, resamples the data to a common sampling rate,
    trims the data to a common time frame, and combines the data into a single pandas DataFrame.

    Parameters:
    folder (str): The path to the folder containing the CSV files.
    desired_sampling_rate (int, optional): The common sampling rate to which all data should be resampled. Default is 64.
    verbose (bool, optional): If True, print additional information during the execution of the function. Default is False.

    Returns:
    pandas.DataFrame: A DataFrame containing the combined data from all CSV files. Each file's data is in a column named
    after the file (with '.csv' removed), except for 'ACC.csv', which is split into three columns: 'acc_x', 'acc_y', and 'acc_z'.
    The DataFrame also includes a 'datetime' column with the time in datetime format, a 'unix_time' column with the time in Unix timestamp format,
    and a 'source' column containing the folder name.

    Note:
    The function expects each CSV file to have the starting time as the first row, the sampling rate as the second row,
    and the signal data starting from the third row. If the signal length is less than 3, the file is skipped.

    Example usage:
    folder = '/path/to/folder'
    desired_sampling_rate = 64
    verbose = True
    combined_data, trimmings = load_data_and_combine(folder, desired_sampling_rate, verbose)
    """

    file_names = ['ACC.csv', 'TEMP.csv', 'EDA.csv', 'BVP.csv', 'HR.csv', 'IBI.csv']
    data_frames = []
    uniqueness_check = pd.DataFrame(columns=['file_path', 'file_name', 'starting_time', 'ending_time'])
    starting_times = []
    ending_times = []

    for file_name in file_names:
        file_path = os.path.join(folder, file_name)
        df = pd.read_csv(file_path, header=None)

        starting_time = df.iloc[0, 0]

        # --- Handle IBI.csv ---
        if file_name == 'IBI.csv':
            # 
            continue

        sampling_rate = int(df.iloc[1, 0])
        signal = df.iloc[2:, :].values
        
        # Check if the signal length is large enough
        if signal.shape[0] < 3:
            print(f"Skipping {file_name} in {folder} due to insufficient signal length.")
            continue
        
        # Resample signal to desired_sampling_rate
        signal_data = nk.signal_resample(signal, sampling_rate=sampling_rate, desired_sampling_rate=desired_sampling_rate, method="FFT")
        
        signal_data = pd.DataFrame(signal_data)

        ending_time = starting_time + signal_data.shape[0] / desired_sampling_rate

        uniqueness_check_child = pd.DataFrame([{'file_path': file_path, 'file_name': file_name, 'starting_time': starting_time, 'ending_time': ending_time}])
        temp_list = [uniqueness_check, uniqueness_check_child]
        uniqueness_check = pd.concat([df for df in temp_list if not df.empty], ignore_index=True).reset_index(drop=True)

        if verbose:
            print(f"File name: {file_name}, starting time: {starting_time}, ending time: {ending_time}, sampling rate: {sampling_rate}, signal shape: {signal.shape}")

        # Set column names
        if file_name == 'ACC.csv':  # Handle the ACC.csv file with 3 columns
            signal_data.columns = ['acc_x', 'acc_y', 'acc_z']
        else:
            column_name = file_name.lower().replace('.csv', '')
            signal_data.columns = [column_name]

        starting_times.append(starting_time)
        ending_times.append(ending_time)
        data_frames.append(signal_data)
    
    # Trim dataframes
    trimmed_data_frames, latest_start_time, earliest_end_time = trim_dataframes(data_frames, starting_times, ending_times, sr=desired_sampling_rate)
    trimmings = max([df.shape[0] for df in data_frames]) - max([df.shape[0] for df in trimmed_data_frames]) # Number of samples trimmed

    if verbose:
        print(f"latest_start_time: {latest_start_time}, earliest_end_time: {earliest_end_time}.")
        print(f"Original longest dataframe length: {max([df.shape[0] for df in data_frames])}, Trimmed dataframe length: {max([df.shape[0] for df in trimmed_data_frames])}. Difference: {trimmings} samples.")
    
    reset_index_data_frames = [df.reset_index(drop=True) for df in trimmed_data_frames]

    concatenated_df = pd.concat(reset_index_data_frames, axis=1)

    # Add time column starting at latest_start_time and ending at earliest_end_time
    time_unix = np.arange(latest_start_time, earliest_end_time, 1 / desired_sampling_rate)
    datetime = pd.to_datetime(time_unix, unit='s', origin='unix')

    # Add time column to the concatenated_df
    concatenated_df['datetime'] = datetime
    concatenated_df['unix_time'] = time_unix

    # Distill file name from folder (e.g. "file" from "input/empatica/file")
    file = folder.split('/')[-1]
    
    concatenated_df['source'] = file

    return concatenated_df, ibi_df, trimmings, uniqueness_check

def trim_dataframes(data_frames, starting_times, ending_times, sr):
    """
    Trim the given data frames based on the starting and ending times.

    Args:
        data_frames (list): A list of pandas DataFrames to be trimmed.
        starting_times (list): A list of starting times for each DataFrame.
        ending_times (list): A list of ending times for each DataFrame.
        sr (int): The sampling rate of the data.

    Returns:
        tuple: A tuple containing the trimmed data frames, the latest start time, and the earliest end time.
    """
    latest_start_time = max(starting_times)
    earliest_end_time = min(ending_times)

    trimmed_data_frames = []
    for idx, df in enumerate(data_frames):
        start_idx = int((latest_start_time - starting_times[idx]) * sr)
        end_idx = int((earliest_end_time - starting_times[idx]) * sr)
        df_trimmed = df.iloc[start_idx:end_idx, :]
        trimmed_data_frames.append(df_trimmed)

    return trimmed_data_frames, latest_start_time, earliest_end_time

