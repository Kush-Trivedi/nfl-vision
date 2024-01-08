# Import Libraries
import pandas as pd
import warnings
import boto3
warnings.filterwarnings("ignore", message="Setting an item of incompatible dtype is deprecated.*")


class NFLDataLoader:
    # bucket_name = "nfl-big-data-bowl-2024"
    # s3_prefix = "assets/"
    path = "assets/nfl-big-data-bowl-2024/"
    
    def __init__(self):
        self.games = None
        self.players = None
        self.plays = None
        self.tackles = None
        self.tracking = None
        self.s3_client = boto3.client('s3')
    
    def downcast_memory_usage(self, df, df_name, verbose=True):
        start_mem = df.memory_usage().sum() / 1024**2
        for col in df.columns:
            dtype_name = df[col].dtype.name
            if dtype_name == 'object':
                pass
            elif dtype_name == 'bool':
                df[col] = df[col].astype('int8')
            elif dtype_name.startswith('int') or (df[col].round() == df[col]).all():
                df[col] = pd.to_numeric(df[col], downcast='integer')
            else:
                df[col] = pd.to_numeric(df[col], downcast='float')
        end_mem = df.memory_usage().sum() / 1024**2

        if verbose:
            print('\033[1;30;34m{}\033[0;0m: Compressed by \033[1m{:.1f}%\033[0m'.format(df_name,100 * (start_mem - end_mem) / start_mem))

        return df

    def load_data(self, file_name):
        # s3_path = f"s3://{self.bucket_name}/{self.s3_prefix}{file_name}"
        # return pd.read_csv(s3_path)
        return pd.read_csv(self.path + file_name)


    def load_all_data(self):
        files = ["games.csv", "players.csv", "plays.csv", "tackles.csv"]
        tracking_files = [f"tracking_week_{week_num}.csv" for week_num in range(1, 10)]
        
        self.games = self.load_data(files[0])
        self.players = self.load_data(files[1])
        self.plays = self.load_data(files[2])
        self.tackles = self.load_data(files[3])
        tracking_data = [self.load_data(file_name) for file_name in tracking_files]
        self.tracking = pd.concat(tracking_data, ignore_index=True)
   
        self.games = self.downcast_memory_usage(self.games, "Games Dataset")
        self.players = self.downcast_memory_usage(self.players, "Players Dataset")
        self.plays = self.downcast_memory_usage(self.plays, "Plays Dataset")
        self.tackles = self.downcast_memory_usage(self.tackles, "Tackles Dataset")
        self.tracking = self.downcast_memory_usage(self.tracking, "Tracking Dataset")
   

    def load_large_file(self, file_path):
        chunk_size = 100000
        chunks = pd.read_csv(file_path, chunksize=chunk_size)
        return pd.concat(chunks, ignore_index=True)
    
    def assign_team(self, row):
        if row['homeTeamAbbr'] == row['club']:
            return f"Home Team - {row['club']}"
        elif row['visitorTeamAbbr'] == row['club']:
            return f"Away Team - {row['club']}"
        else:
            return 'Football'
    
    def get_overall_plays_with_tracking(self):
        plays_with_tracking = pd.merge(self.tracking, self.plays, on=['gameId', 'playId'], how='inner')
        ball_id = 999999 
        tackle_plays_with_tracking = pd.merge(plays_with_tracking, self.tackles, on=['gameId', 'playId', 'nflId'], how='left')
        merged_data = pd.merge(tackle_plays_with_tracking, self.players, on=['nflId','displayName'], how='left')
        merged_data['nflId'].fillna(ball_id, inplace=True)
        merged_data['jerseyNumber'].fillna("", inplace=True)  
        merged_data.rename(columns={'club': 'Team'}, inplace=True)
        return merged_data

    def basic_summary(self, data_frame, data_set_name):
        summary = pd.DataFrame(data_frame.dtypes, columns=['Data Type'])
        summary = summary.reset_index()
        summary = summary.rename(columns={'index': 'Feature'})
        summary['Num of Nulls'] = data_frame.isnull().sum().values
        summary['Num of Unique'] = data_frame.nunique().values
        summary['First Value'] = data_frame.iloc[0].values
        summary['Second Value'] = data_frame.iloc[1].values
        summary['Third Value'] = data_frame.iloc[2].values
        summary['Fourth Value'] = data_frame.iloc[3].values
        return summary