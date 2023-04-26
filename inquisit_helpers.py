import pandas as pd

def load_and_process(filepath):
    df = pd.read_csv(filepath, sep="\t")

    # Select all rows where `blockcode` contains `TNT`
    df = df[df['blockcode'].str.contains("TNT_NoThink")]

    # Add time column by combining `date` and `currenttime_plus_ms` columns into a datetime object column with milliseconds
    df['time'] = pd.to_datetime(df['date'] + " " + df['currenttime_plus_ms'], format="%Y-%m-%d %H:%M:%S:%f")

    # Remove cols date, currenttime, currenttime_plus_ms
    df = df.drop(columns=['date', 'currenttime', 'currenttime_plus_ms'])
    df = df[['time', 'blockcode', 'trialcode', 'response', 'blocknum']]

    # Group df by trialnum
    grouped = df.groupby('blocknum')
    # Loop over each groups
    for name, group in grouped:
        response = group[group['trialcode'] == "Intrusion"].iloc[0]['response']
        df.loc[(df['blocknum'] == name) & (df['trialcode'].str.contains("TNT_NoThink")), 'response'] = response

    # Remove all rows where trialcode does not contain "TNT_NoThink"
    df = df[df['trialcode'].str.contains("TNT_NoThink")]
    df = df.drop(columns=['blocknum', 'blockcode'])

    return df 