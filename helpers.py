import os 
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def get_dir_list(path):
    """
    This function returns a list of all the directories in the given path.
    
    :param path: The path to search for directories.
    :return: A list of directories.
    """
    return [x for x in os.listdir(path) if x != ".DS_Store"]

def merge_inquisit_empatica(empatica_df, inquisit_df):
    """
    This function merges the given empatica and inquisit DataFrames.

    :param empatica_df: The empatica DataFrame.
    :param inquisit_df: The inquisit DataFrame.
    :return: A merged DataFrame.
    """
    empatica_df['time'] = empatica_df['time'].astype('datetime64[ns]').astype(int) / 10**9
    inquisit_df['time'] = inquisit_df['time'].astype('datetime64[ns]').astype(int) / 10**9

    # Sort both DataFrames by the 'time' column
    empatica_df = empatica_df.sort_values('time')
    inquisit_df = inquisit_df.sort_values('time')

    # Merge the two DataFrames using merge_asof on the 'time' column and rename the 'time' column from inquisit_df to 'time_y'
    merged_df = pd.merge_asof(empatica_df, inquisit_df[['time', 'response', 'trialcode']].rename(columns={'time': 'time_y'}), left_on='time', right_on='time_y', direction='nearest')

    merged_df['time'] = pd.to_datetime(merged_df['time'], unit='s')
    merged_df['time_y'] = pd.to_datetime(merged_df['time_y'], unit='s')

    # Calculate the time differences and store them in a new 'time_diff' column
    merged_df['time_diff'] = (merged_df['time'] - merged_df['time_y']).abs()

    # Add a new 'response' column to merged_df containing the response value only if the time difference is less than or equal to 0.25 seconds
    merged_df['new_response'] = merged_df.apply(lambda row: row['response'] if row['time_diff'] <= pd.Timedelta(seconds=0.25) else 0, axis=1)

    # Add a new 'trialcode' column to merged_df containing the trialcode value only if the time difference is less than or equal to 0.25 seconds, otherwise set it to NaN
    merged_df['trialcode'] = merged_df.apply(lambda row: row['trialcode'] if row['time_diff'] <= pd.Timedelta(seconds=0.25) else None, axis=1)

    # Drop the unnecessary 'time_y', 'response' and 'time_diff' columns
    merged_df.drop(columns=['time_y', 'response', 'time_diff'], inplace=True)

    # Rename the 'new_response' column to 'response'
    merged_df.rename(columns={'new_response': 'response'}, inplace=True)

    # If the response was 3 or 80, set it to 1; if the response was 4 or 81, set it to 2.
    merged_df['response'] = merged_df['response'].apply(lambda x: 1 if x in [3, 80] else 2 if x in [4, 81] else 0)

    # Add a new column named 'intrusion' containing 1 if the response is 1 or 2, and 0 otherwise 
    merged_df['intrusion'] = merged_df['response'].apply(lambda x: 1 if x in [1, 2] else 0)

    return merged_df

def get_inquisit_tags(filepath):
    """
    This function returns a list of all the timestamps of the inquisit tags in the given file.

    :param filepath: The path to the file to read.
    :return: A list of timestamps.
    """
    df = pd.read_csv(filepath, sep="\t")

    # Select all rows where `blockcode` contains `TNT`
    df = df[(df['blockcode'].str.contains("countdown")) & (df['trialcode'].str.contains("press_button"))]

    # Add time column by combining `date` and `currenttime_plus_ms` columns into a datetime object column with milliseconds
    df['time'] = pd.to_datetime(df['date'] + " " + df['currenttime_plus_ms'], format="%Y-%m-%d %H:%M:%S:%f") - timedelta(hours = 2)

    # Convert 'time' column back to UNIX timestamp
    df['time'] = df['time'].astype('datetime64[ns]').astype(int) / 10**9

    return list(df['time'])

def closest_timestamp(empatica_ts, inquisit_timestamps):
    """
    Find the closest Inquisit timestamp for a given Empatica timestamp.
    
    Args:
        empatica_ts (float): The Empatica timestamp.
        inquisit_timestamps (list): List of Inquisit timestamps.
    
    Returns:
        float: The closest Inquisit timestamp.
    """
    return min(inquisit_timestamps, key=lambda x: abs(x - empatica_ts))

def prepare_datasets(filename, test_size=0.2, val_size=0.1, oversample_ratio=0.25, undersample_ratio=0.25, oversample=False):
    # Data loading
    df = pd.read_csv(filename)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.drop(['datetime', 'unix_time', 'source', 'response', 'intrusion', 'intrusion_nothink', 'trialcode', 'session_id'], axis=1, inplace=True)

    participants = df['participant']
    X = df.drop(['intrusion_tnt', 'participant'], axis=1)  # Features: All columns except 'intrusion_tnt' and 'participant'
    y = df['intrusion_tnt']  # Labels: 'intrusion_tnt' column

    # Create train/test split
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=participants))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Create train/validation split
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=42)
    train_wval_idx, val_idx = next(gss.split(X_train, y_train, groups=participants.iloc[train_idx]))
    X_train, X_val = X_train.iloc[train_wval_idx], X_train.iloc[val_idx]
    y_train, y_val = y_train.iloc[train_wval_idx], y_train.iloc[val_idx]

    # Impute missing values
    knn_imputer = KNNImputer(n_neighbors=5)
    knn_imputer.fit(X_train)
    X_train = pd.DataFrame(knn_imputer.transform(X_train), columns=X_train.columns)
    X_val = pd.DataFrame(knn_imputer.transform(X_val), columns=X_val.columns)
    X_test = pd.DataFrame(knn_imputer.transform(X_test), columns=X_test.columns)

    if oversample:
        # Define the resampling strategy
        over = SMOTE(random_state=42, sampling_strategy=oversample_ratio)
        under = RandomUnderSampler(sampling_strategy=undersample_ratio)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        # Fit the SMOTE instance on the training data
        X_train, y_train = pipeline.fit_resample(X_train, y_train)

    # Normalize the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    # Recombine the training and validation sets for cross-validation
    X_train_val = np.concatenate((X_train, X_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    feature_names = X.columns.tolist()

    # Print lengths of the datasets
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Training + Validation set: {len(X_train_val)} samples")
    print(f"Test set: {len(X_test)} samples")

    return X_train, y_train, X_val, y_val, X_test, y_test, X_train_val, y_train_val, train_idx, train_wval_idx, val_idx, test_idx, df, feature_names, participants

def create_folds(X_train, y_train, groups, n_folds=10):
    """
    Create folds for cross-validation using GroupKFold.

    Parameters:
    - X_train (array-like): The input features for training.
    - y_train (array-like): The target variable for training.
    - groups (array-like): The groups to be used for grouping the samples.
    - n_folds (int): The number of folds to create (default=10).

    Returns:
    - folds (list): A list of tuples containing train and test indices for each fold.
    """

    # Create GroupKFold object
    gkf = GroupKFold(n_splits=n_folds)

    # Folds must be a list of tuples of train and test indices
    folds = []
    for train_index, test_index in gkf.split(X_train, y_train, groups):
        folds.append((train_index.tolist(), test_index.tolist()))

    # Print length of each sublist
    print("Folds created:")
    for fold in folds:
        print(f"Train: {len(fold[0])}, Eval: {len(fold[1])}")

    return folds

def get_eval(y_test, y_pred):
    """
    Calculate evaluation metrics for binary classification.

    Parameters:
    - y_test (array-like): True labels.
    - y_pred (array-like): Predicted labels.

    Returns:
    - report (dict): Dictionary containing evaluation metrics.
    - string (str): String representation of evaluation metrics.
    """
    f1_micro = f1_score(y_test, y_pred, average='micro')
    aucpr = average_precision_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    report = {
        'F1 Score (micro)': f1_micro,
        'AUCPR': aucpr,
        'AUC': auc
    }
    string = f'F1 Score (micro): {f1_micro:.2f}\nAUCPR: {aucpr:.2f}\nAUC: {auc:.2f}'
    return report, string

def xgb_micro_f1(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(preds)
    return 'micro_f1', -f1_score(labels, preds, average='micro')

def xgb_aucpr(preds, dtrain):
    labels = dtrain.get_label()
    return 'aucpr', -average_precision_score(labels, preds)