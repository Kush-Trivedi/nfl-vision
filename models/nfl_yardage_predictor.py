import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import StackingRegressor, VotingRegressor


class NFLDataLoader:
    # bucket_name = "nfl-big-data-bowl-2024"
    # s3_prefix = "assets/"
    path = "../assets/nfl-big-data-bowl-2024/"
    
    def __init__(self):
        self.games = None
        self.players = None
        self.plays = None
        self.tackles = None
        self.tracking = None
        # self.s3_client = boto3.client('s3')
    
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

class NFLYardagePredictor:
    def __init__(self, X, y, groups):
        self.X = X
        self.y = y
        self.groups = groups
        self.group_kfold = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=1205)
        self.preprocessor = self.create_preprocessor()
        self.pipelines = {
            'cat_boost': self.create_pipeline(CatBoostRegressor(silent=True)),
            'xg_boost': self.create_pipeline(XGBRegressor()),
            'lgbm': self.create_pipeline(LGBMRegressor(force_col_wise=True,verbosity=-1)),
            'voting': self.create_voting_pipeline(),
            'stacking': self.create_stacking_pipeline()
        }
    
    @staticmethod
    def calculate_metrics(true, predicted, n, k):
        mse = metrics.mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        mae = metrics.mean_absolute_error(true, predicted)
        r2_square = metrics.r2_score(true, predicted)
        adjusted_r2 = 1 - (((1 - r2_square) * (n - 1)) / (n - k - 1))

        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2 Square': r2_square, 'Adjusted R2': adjusted_r2}
    
    @staticmethod
    def evaluate(true, predicted, n, k):  
        metrics_result = NFLYardagePredictor.calculate_metrics(true, predicted, n, k)

        metrics_df = pd.DataFrame({
            'Metric': ['MAE', 'MSE', 'RMSE', 'R2 Square', 'Adjusted R2'],
            'Value': [
                round(metrics_result['MAE'], 4),
                round(metrics_result['MSE'], 4),
                round(metrics_result['RMSE'], 4),
                f'{round(metrics_result["R2 Square"] * 100, 4)}%',
                f'{round(metrics_result["Adjusted R2"] * 100, 4)}%'
            ]
        })

        # Print the DataFrame
        print(metrics_df.to_string(index=False))
        print(20 * '=', '\n')

        return metrics_result
    
    @staticmethod
    def print_metrics(metric_dict):
        metrics_df = pd.DataFrame(metric_dict.items(), columns=['Metric', 'Value'])
        print(metrics_df.to_string(index=False))
        print(20 * '=', '\n')

    def create_preprocessor(self):
        categorical_cols = [cname for cname in self.X.columns if self.X[cname].dtype == "object"]
        numerical_cols = [cname for cname in self.X.columns if self.X[cname].dtype in ['int64', 'float64']]
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('std_scaler', MinMaxScaler())
        ])
        return ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )

    def create_pipeline(self, model):
        return Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('model', model)
        ])

    def create_voting_pipeline(self):
        cat_boost = CatBoostRegressor(silent=True)
        xg_boost = XGBRegressor()
        lgbm = LGBMRegressor(force_col_wise=True,verbosity=-1)
        voting = VotingRegressor(estimators=[('LGBM', lgbm), ('XGB', xg_boost), ('CAT', cat_boost)], weights=[1, 2, 2])
        return self.create_pipeline(voting)

    def create_stacking_pipeline(self):
        lgbm = LGBMRegressor(force_col_wise=True,verbosity=-1)
        xg_boost = XGBRegressor()
        cat_boost = CatBoostRegressor(silent=True)
        layer_one_estimators = [('LGBM', lgbm), ('CAT', cat_boost)]
        layer_two_estimators = [('XGB', xg_boost), ('CAT', cat_boost)]
        layers = StackingRegressor(estimators=layer_two_estimators, final_estimator=xg_boost)
        stacking = StackingRegressor(estimators=layer_one_estimators, final_estimator=layers)
        return self.create_pipeline(stacking)

    def train_evaluate(self, pipeline_key, validation_size=0.3):
        all_results = []
        oof_predictions = np.zeros(len(self.X))
        metrics_summary = {'MAE': [], 'MSE': [], 'RMSE': [], 'R2 Square': [], 'Adjusted R2': []}
        
        n = len(self.X)  
        k = len(self.X.columns) 

        for fold, (train_index, test_index) in enumerate(self.group_kfold.split(self.X, self.y, self.groups)):
            print(20 * '=')
            print(f"Processing Fold {fold + 1}...")
            print(20 * '=', '\n')
          
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)

            pipeline = self.pipelines[pipeline_key]
            pipeline.fit(X_train, y_train)


            val_pred = pipeline.predict(X_val)
            print(f"Validation - Fold {fold + 1}")
            val_metrics = self.evaluate(y_val, val_pred, len(X_val), k)
            
            test_pred = pipeline.predict(X_test)
            print(f"Test - Fold {fold + 1}")
            test_metrics = self.evaluate(y_test, test_pred, len(X_test), k)
            oof_predictions[test_index] = test_pred

            for key in test_metrics:
                metrics_summary[key].append(test_metrics[key])

            fold_results = pd.DataFrame({
                # "gameId": X_test['gameId'],
                # "playId": X_test['playId'],
                "Actual Yardage": y_test,
                "Expected Yardage": test_pred
            })
            all_results.append(fold_results)

        # Calculate and print OOF metrics
        oof_metrics = self.calculate_metrics(self.y, oof_predictions, n, k)  # Use a method that doesn't print
        print(20 * '=')
        print("OOF Metrics:")
        print(20 * '=', '\n')
        self.print_metrics(oof_metrics)

        # Calculate and print mean of all metrics across folds
        mean_metrics = {key: np.mean(metrics_summary[key]) for key in metrics_summary}
        print(22 * '=')
        print("Mean Across All Folds:")
        print(22 * '=', '\n')
        self.print_metrics(mean_metrics)

        final_results_df = pd.concat(all_results, ignore_index=True)
        final_results_df["Expected Yardage"] = final_results_df["Expected Yardage"].round().astype(int)
        print(final_results_df)
        
    def save_model(self, pipeline_key, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.pipelines[pipeline_key], file)
            
# Load Required Data
nfl_data_loader = NFLDataLoader()
tracking_files = [f"tracking_week_{week_num}.csv" for week_num in range(1, 10)]
tracking_data = [nfl_data_loader.load_data(file_name) for file_name in tracking_files]
tracking_df = pd.concat(tracking_data, ignore_index=True)
plays_df = nfl_data_loader.load_data("plays.csv")
players_df = nfl_data_loader.load_data("players.csv")

plays_with_tracking = pd.merge(tracking_df, plays_df, on=['gameId', 'playId'], how='inner')
players_plays_with_tracking = pd.merge(plays_with_tracking, players_df, on=['nflId','displayName'], how='inner')

categorical_columns = ['club','possessionTeam','defensiveTeam', 'position','event','offenseFormation','ballCarrierDisplayName'] 
label_encoders = {}
for column in categorical_columns:
    if column in players_plays_with_tracking.columns:
        label_encoders[column] = LabelEncoder()
        players_plays_with_tracking[column] = label_encoders[column].fit_transform(players_plays_with_tracking[column].astype(str))


df = players_plays_with_tracking[['gameId','playId','club','quarter', 'down', 'yardsToGo', 'possessionTeam', 'event',
       'defensiveTeam', 'yardlineNumber', 'absoluteYardlineNumber', 'offenseFormation',
       'defendersInTheBox', 'passProbability', 'preSnapHomeTeamWinProbability',
       'preSnapVisitorTeamWinProbability', 'homeTeamWinProbabilityAdded',
       'visitorTeamWinProbilityAdded', 'expectedPoints', 'expectedPointsAdded',
       'weight','ballCarrierDisplayName', 'position','playResult']].copy()


df['YardlineAdvantage'] = (100 - df['yardlineNumber']) * df['expectedPointsAdded']

df = df[['gameId','playId','club','quarter', 'down', 'yardsToGo', 'possessionTeam', 'event',
       'defensiveTeam', 'yardlineNumber', 'absoluteYardlineNumber', 'offenseFormation',
       'defendersInTheBox', 'passProbability', 'preSnapHomeTeamWinProbability',
       'preSnapVisitorTeamWinProbability', 'homeTeamWinProbabilityAdded',
       'visitorTeamWinProbilityAdded', 'expectedPoints', 'expectedPointsAdded',
       'weight','ballCarrierDisplayName', 'position','YardlineAdvantage','playResult']]


# Compute the correlation matrix
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})

# prepare final data 
df = (df[["event", "down", "yardsToGo","yardlineNumber","defendersInTheBox","passProbability","expectedPoints","YardlineAdvantage","playResult"]]
      .dropna(subset=['event'])
      .drop_duplicates()
      .reset_index(drop=True))

groups = df['event']
X = df.drop(['playResult', 'event'], axis=1) 
y = df['playResult']

nfl_yardage_pipeline = NFLYardagePredictor(X, y, groups)
nfl_yardage_pipeline.train_evaluate('stacking') 
nfl_yardage_pipeline.save_model('stacking', 'stacking_model.pkl')