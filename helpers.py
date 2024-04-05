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
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, roc_auc_score, auc
import matplotlib.pyplot as plt
import seaborn as sns
import neurokit2 as nk

def get_dir_list(path):
    """
    This function returns a list of all the directories in the given path.
    
    :param path: The path to search for directories.
    :return: A list of directories.
    """
    return [x for x in os.listdir(path) if x != ".DS_Store"]

def combine_empatica_and_inquisit(empatica_df, inquisit_df, save=False):
    # Add unix_time to df (inquisit_df['time'][0] = 2023-04-18 16:59:11.535)
    inquisit_df['unix_time'] = pd.to_datetime(inquisit_df['time']).astype(int) / 10**9

    # Rename time to datetime
    inquisit_df = inquisit_df.rename(columns={"time": "datetime"})

    # Sort both DataFrames by the 'time' column
    empatica_df = empatica_df.sort_values('unix_time')
    inquisit_df = inquisit_df.sort_values('unix_time')

    merged_df = pd.merge_asof(empatica_df, inquisit_df, on="unix_time", direction="nearest", tolerance=1)
    merged_df['datetime_x'] = pd.to_datetime(merged_df['datetime_x'])
    merged_df['datetime_y'] = pd.to_datetime(merged_df['datetime_y'])
    merged_df['delta_t'] = (merged_df['datetime_x'] - merged_df['datetime_y']).abs()

    # Add a new 'response' column to merged_df containing the response value only if the time difference is less than or equal to 0.25 seconds
    merged_df['new_response'] = merged_df.apply(lambda row: row['response'] if row['delta_t'] <= pd.Timedelta(seconds=1/64) else 0, axis=1)

    # Add a new 'trialcode' column to merged_df containing the trialcode value only if the time difference is less than or equal to 0.25 seconds, otherwise set it to NaN
    merged_df['trialcode'] = merged_df.apply(lambda row: row['trialcode'] if row['delta_t'] <= pd.Timedelta(seconds=1/64) else None, axis=1)

    merged_df.drop(columns=['datetime_y', 'response', 'delta_t'], inplace=True)
    merged_df.rename(columns={"datetime_x": "datetime"}, inplace=True)
    merged_df.rename(columns={'new_response': 'response'}, inplace=True)

    # If the response was 3 or 80, set it to 1; if the response was 4 or 81, set it to 2.
    merged_df['response'] = merged_df['response'].apply(lambda x: 1 if x in [3, 80] else 2 if x in [4, 81] else 0)

    # Add a new column named 'intrusion' containing 1 if the response is 1 or 2, and 0 otherwise 
    merged_df['intrusion'] = merged_df.apply(lambda row: 1 if row['response'] in [1, 2] and row['trialcode'] in ["TNT_NoThink_CSm", "TNT_NoThink_CSp"] else 0, axis=1)

    merged_df['intrusion_nothink'] = merged_df.apply(lambda row: 1 if row['response'] in [1, 2] and row['trialcode'] in ["TNT_NoThink_CSm", "TNT_NoThink_CSp"] else 0 if row['trialcode'] in ["TNT_NoThink_CSm", "TNT_NoThink_CSp"] else None, axis=1)
    merged_df['intrusion_tnt'] = merged_df.apply(lambda row: 1 if row['response'] in [1, 2] and row['trialcode'] in ["TNT_NoThink_CSm", "TNT_NoThink_CSp"] else 0 if pd.notna(row['trialcode']) else None, axis=1)

    # Create a dictionary that maps the original filenames to the new filenames
    filename_map = {
        'd1 2': 'pp15_d1 2',
        'd1_1': 'pp13_d1_1',
        '1681713254_A03F6E': 'pp16_1681713254_A03F6E',
        '1681717717_A03F6E': 'pp17_1681717717_A03F6E',
        'd1_3': 'pp18_d1_3',
        'd2_1_1': 'pp17_d2_1_1',
        'd2_2': 'pp16_d2_2',
        'd1': 'pp19_d1',
        'd1_4': 'pp20_d1_4',
        'd2': 'pp18_d2',
        'd2_1': 'pp19_d2_1',
        'd2_4': 'pp20_d2_4'
    }

    # Update the 'source' column
    merged_df['source'] = merged_df['source'].replace(filename_map)

    merged_df['participant'] = merged_df['source'].str.extract('pp(\d{1,2})').astype(float)

    # Rename values for participant to be incremental integers starting from 1
    merged_df['participant'] = merged_df['participant'].replace({3: 1, 2: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 13: 11, 15: 12, 16: 13, 17: 14, 18: 15, 19: 16, 20: 17})

    # Turn to int
    merged_df['participant'] = merged_df['participant'].astype(int)

    if save:
        merged_df.to_csv('output/empatica_inquisit_merged.csv', index=False)
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

def preprocess_data(df = None, filename='output/empatica_inquisit_merged.csv', save=False, sr=64):
    print("Preprocessing data...")
    # Data loading
    if df is None:
        df = pd.read_csv(filename)
    
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Filtering 
    df['eda'] = nk.eda_clean(df['eda'], sampling_rate=64, method='biosppy')
    df['bvp'] = nk.ppg_clean(df['bvp'], sampling_rate=64, heart_rate=None, method='elgendi')

    # Calculate body acceleration
    df['body_acc'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

    # Drop acc columns
    df = df.drop(columns=['acc_x', 'acc_y', 'acc_z'])

    columns_to_normalize = ["body_acc", "temp", "eda", "bvp", "hr"]
    participants = df['participant']

    # Apply StandardScaler to each participant group and avoid resetting the index
    scaled_df = df.groupby(participants)[columns_to_normalize].apply(lambda x: pd.DataFrame(StandardScaler().fit_transform(x), columns=x.columns) if len(x) > 1 else x)

    # Reset the inner level of the index
    scaled_df.reset_index(drop=True, inplace=True)

    # Merge the scaled data back into the original dataframe
    df[columns_to_normalize] = scaled_df
    print("Normalised columns: ", columns_to_normalize)

    # Create a column 'session_id' that identifies each session
    df['session_id'] = (df['datetime'].diff() > pd.Timedelta(seconds=1/64)).cumsum()
    # Create a column 'session_duration' that indicates the duration of each session
    df['session_duration'] = df.groupby('session_id')['datetime'].transform(lambda x: x.max() - x.min())
    # Drop all sessions that are shorter than 8 seconds
    print(f"Number of sessions before filtering: {len(df['session_id'].unique())}")
    df = df[df['session_duration'].dt.total_seconds() > 8]
    print(f"Number of sessions after filtering: {len(df['session_id'].unique())}")

    df.drop(columns=['session_duration'])

    # EDA
    signals, info = nk.eda_process(df['eda'], sampling_rate=sr)
    df['eda'] = signals['EDA_Clean'].values
    df.drop(columns=['eda'])

    df['eda_tonic'] = signals['EDA_Tonic'].values
    df['eda_phasic'] = signals['EDA_Phasic'].values

    if save:
        df.to_csv('output/ei_prep.csv', index=False)
    
    print("Preprocessing complete.")
    return df

def get_features(df = None, filename='output/ei_prep.csv', s=8, sr=64, save=False, reference_class='tnt'):
    print('Engineering features...')
    # Meant for 'output/combined_feature_engineered_tnt_only.csv'
    # Data loading
    if df is None:
        df = pd.read_csv(filename)
    
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Calculate the window size in samples
    win = sr*s

    # Group by 'session_id'
    temp = df.groupby('session_id')

    # EDA
    # EDA Tonic
    df['eda_tonic_mean'] = temp['eda_tonic'].rolling(window=win).mean().reset_index(0, drop=True)
    df['eda_tonic_std'] = temp['eda_tonic'].rolling(window=win).std().reset_index(0, drop=True)
    df['eda_tonic_min'] = temp['eda_tonic'].rolling(window=win).min().reset_index(0, drop=True)
    df['eda_tonic_max'] = temp['eda_tonic'].rolling(window=win).max().reset_index(0, drop=True)
    df['eda_tonic_skew'] = temp['eda_tonic'].rolling(window=win).skew().reset_index(0, drop=True)
    df['eda_tonic_kurt'] = temp['eda_tonic'].rolling(window=win).kurt().reset_index(0, drop=True)

    # EDA Phasic
    df['eda_phasic_mean'] = temp['eda_phasic'].rolling(window=win).mean().reset_index(0, drop=True)
    df['eda_phasic_std'] = temp['eda_phasic'].rolling(window=win).std().reset_index(0, drop=True)
    df['eda_phasic_min'] = temp['eda_phasic'].rolling(window=win).min().reset_index(0, drop=True)
    df['eda_phasic_max'] = temp['eda_phasic'].rolling(window=win).max().reset_index(0, drop=True)
    df['eda_phasic_skew'] = temp['eda_phasic'].rolling(window=win).skew().reset_index(0, drop=True)
    df['eda_phasic_kurt'] = temp['eda_phasic'].rolling(window=win).kurt().reset_index(0, drop=True)

    # ACCELEROMETER
    df['body_acc_mean'] = temp['body_acc'].rolling(window=win).mean().reset_index(0, drop=True)
    df['body_acc_std'] = temp['body_acc'].rolling(window=win).std().reset_index(0, drop=True)
    df['body_acc_min'] = temp['body_acc'].rolling(window=win).min().reset_index(0, drop=True)
    df['body_acc_max'] = temp['body_acc'].rolling(window=win).max().reset_index(0, drop=True)
    df['body_acc_skew'] = temp['body_acc'].rolling(window=win).skew().reset_index(0, drop=True)
    df['body_acc_kurt'] = temp['body_acc'].rolling(window=win).kurt().reset_index(0, drop=True)

    # TEMPERATURE
    df['temp_mean'] = temp['temp'].rolling(window=win).mean().reset_index(0, drop=True)
    df['temp_std'] = temp['temp'].rolling(window=win).std().reset_index(0, drop=True)
    df['temp_min'] = temp['temp'].rolling(window=win).min().reset_index(0, drop=True)
    df['temp_max'] = temp['temp'].rolling(window=win).max().reset_index(0, drop=True)
    df['temp_skew'] = temp['temp'].rolling(window=win).skew().reset_index(0, drop=True)
    df['temp_kurt'] = temp['temp'].rolling(window=win).kurt().reset_index(0, drop=True)

    # HEART RATE
    df['hr_mean'] = temp['hr'].rolling(window=win).mean().reset_index(0, drop=True)
    df['hr_std'] = temp['hr'].rolling(window=win).std().reset_index(0, drop=True)
    df['hr_min'] = temp['hr'].rolling(window=win).min().reset_index(0, drop=True)
    df['hr_max'] = temp['hr'].rolling(window=win).max().reset_index(0, drop=True)
    df['hr_skew'] = temp['hr'].rolling(window=win).skew().reset_index(0, drop=True)
    df['hr_kurt'] = temp['hr'].rolling(window=win).kurt().reset_index(0, drop=True)

    # BVP
    df['bvp_mean'] = temp['bvp'].rolling(window=win).mean().reset_index(0, drop=True)
    df['bvp_std'] = temp['bvp'].rolling(window=win).std().reset_index(0, drop=True)
    df['bvp_min'] = temp['bvp'].rolling(window=win).min().reset_index(0, drop=True)
    df['bvp_max'] = temp['bvp'].rolling(window=win).max().reset_index(0, drop=True)
    df['bvp_skew'] = temp['bvp'].rolling(window=win).skew().reset_index(0, drop=True)
    df['bvp_kurt'] = temp['bvp'].rolling(window=win).kurt().reset_index(0, drop=True)

    if reference_class == 'tnt':
        # Only keep rows that dont have nan for intrusion_tnt 
        df = df.dropna(subset=['intrusion_tnt'])
        # Rename intrusion_tnt to intrusion
        df = df.drop(columns=['intrusion_nothink', 'intrusion'])
        df = df.rename(columns={'intrusion_tnt': 'intrusion'})
    elif reference_class == 'nt':
        # Only keep rows that dont have nan for intrusion_nothink
        df = df.dropna(subset=['intrusion_nothink'])
        # Rename intrusion_nothink to intrusion
        df = df.drop(columns=['intrusion_tnt', 'intrusion'])
        df = df.rename(columns={'intrusion_nothink': 'intrusion'})
    elif reference_class == 'all':
        # Keep all rows
        df = df.drop(columns=['intrusion_nothink', 'intrusion_tnt'])
    else:
        raise ValueError("reference_class must be 'tnt' or 'nt'")

    df = df.drop(columns=['temp', 'hr', 'eda', 'body_acc', 'bvp', 'eda_tonic', 'eda_phasic'])
    df = df.drop(columns=['session_id', 'session_duration'])

    if save:
        df.to_csv(f'output/dataset_{reference_class}_win{s}.csv', index=False)

    print('Feature engineering done!')
    return df
    
def prepare_datasets(df = None, filename = 'output/combined_feature_engineered_tnt_only_win8.csv', test_size=0.2, val_size=0.1):
    print("Preparing datasets...")
    if df is None:
        df = pd.read_csv(filename)

    df['datetime'] = pd.to_datetime(df['datetime'])
    df.drop(['datetime', 'unix_time', 'source', 'response', 'trialcode'], axis=1, inplace=True)

    participants = df['participant']
    X = df.drop(['intrusion', 'participant'], axis=1)  # Features: All columns except 'intrusion_tnt' and 'participant'
    y = df['intrusion']  # Labels: 'intrusion_tnt' column

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

    # Recombine the training and validation sets for cross-validation
    X_train_val = np.concatenate((X_train, X_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)

    feature_names = X.columns.tolist()

    # Print lengths of the datasets
    print(f"Training set: {len(X_train)} rows of features, {len(y_train)} labels")
    print(f"Validation set: {len(X_val)} rows of features, {len(y_val)} labels")
    print(f"Training + Validation set: {len(X_train_val)} rows of features, {len(y_train_val)} labels")
    print(f"Test set: {len(X_test)} rows of features, {len(y_test)} labels")

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

def xgb_micro_f1(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(preds)
    return 'micro_f1', -f1_score(labels, preds, average='micro')

def xgb_aucpr(preds, dtrain):
    labels = dtrain.get_label()
    return 'aucpr', -average_precision_score(labels, preds)

def plot_metrics(y_val, y_pred, model, X_val):
    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xticks([0.5, 1.5], ['0', '1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # AUPRC curve
    precision, recall, thresholds = precision_recall_curve(y_val, model.predict_proba(X_val)[:,1])
    auc_score = auc(recall, precision)
    plt.plot(recall, precision, marker='.', label='XGBoost')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve: AUC = {:.2f}'.format(auc_score))
    plt.legend()
    plt.show()

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_val, model.predict_proba(X_val)[:,1])
    roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:,1])
    plt.plot(fpr, tpr, marker='.', label='XGBoost')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: AUC = {:.2f}'.format(roc_auc))
    plt.legend()
    plt.show()
