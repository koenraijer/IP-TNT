import os 
import pandas as pd

def get_dir_list(path):
    """
    This function returns a list of all the directories in the given path.
    
    :param path: The path to search for directories.
    :return: A list of directories.
    """
    return [x for x in os.listdir(path) if x != ".DS_Store"]

def merge_inquisit_empatica(empatica_df, inquisit_df):
    empatica_df['time'] = empatica_df['time'].astype('datetime64[ns]').astype(int) / 10**9
    inquisit_df['time'] = inquisit_df['time'].astype('datetime64[ns]').astype(int) / 10**9

    # Sort both DataFrames by the 'time' column
    empatica_df = empatica_df.sort_values('time')
    inquisit_df = inquisit_df.sort_values('time')

    # Merge the two DataFrames using merge_asof on the 'time' column and rename the 'time' column from inquisit_df to 'time_y'
    merged_df = pd.merge_asof(empatica_df, inquisit_df[['time', 'response']].rename(columns={'time': 'time_y'}), left_on='time', right_on='time_y', direction='nearest')

    merged_df['time'] = pd.to_datetime(merged_df['time'], unit='s')
    merged_df['time_y'] = pd.to_datetime(merged_df['time_y'], unit='s')

    # Calculate the time differences and store them in a new 'time_diff' column
    merged_df['time_diff'] = (merged_df['time'] - merged_df['time_y']).abs()

    # Add a new 'response' column to merged_df containing the response value only if the time difference is less than or equal to 0.25 seconds
    merged_df['new_response'] = merged_df.apply(lambda row: row['response'] if row['time_diff'] <= pd.Timedelta(seconds=0.25) else 0, axis=1)

    # Drop the unnecessary 'time_y' and 'response' columns
    merged_df.drop(columns=['time_y', 'response'], inplace=True)

    # Rename the 'new_response' column to 'response'
    merged_df.rename(columns={'new_response': 'response'}, inplace=True)

    # If the response was 3 or 80, set it to 1; if the response was 4 or 81, set it to 2.
    merged_df['response'] = merged_df['response'].apply(lambda x: 1 if x in [3, 80] else 2 if x in [4, 81] else 0)

    # Add a new column named 'intrusion' containing 1 if the response is 1 or 2, and 0 otherwise 
    merged_df['intrusion'] = merged_df['response'].apply(lambda x: 1 if x in [1, 2] else 0)

    return merged_df