# Import Libraries
import pandas as pd

class NFLStats:
    def __init__(self, games_df, player_df, tackles_df):
        self.games_df = games_df
        self.player_df = player_df
        self.tackles_df = tackles_df

    def calculate_player_stats(self):
        player_tackles_df = pd.merge(self.player_df, self.tackles_df, on="nflId", how="inner")
        player_tackles_games_df = pd.merge(player_tackles_df,self.games_df, on="gameId", how="left")
        player_stats = player_tackles_games_df.groupby(["displayName"]).agg(
            Season=pd.NamedAgg(column="season", aggfunc="first"),
            total=pd.NamedAgg(column="gameId", aggfunc="size"),
            tackle=pd.NamedAgg(column="tackle", aggfunc="sum"),
            assist=pd.NamedAgg(column="assist", aggfunc="sum"),
            forced_fumble=pd.NamedAgg(column="forcedFumble", aggfunc="sum"),
            missed_tackle=pd.NamedAgg(column="pff_missedTackle", aggfunc="sum")
        ).reset_index()
        
        percentage_columns = ['tackle', 'assist', 'forced_fumble', 'missed_tackle']
        for col in percentage_columns:
            player_stats[f'{col}_accuracy'] = (player_stats[col] / player_stats['total'] * 100).round(2)
        
        player_stats['overall_accuracy'] = (
            (player_stats['tackle_accuracy'] * player_stats['tackle'] + 
             player_stats['assist_accuracy'] * player_stats['assist'] +
             player_stats['forced_fumble_accuracy'] * player_stats['forced_fumble'] -
             player_stats['missed_tackle_accuracy'] * player_stats['missed_tackle']
            ) / player_stats['total']
        ).round(2)
        
        player_stats['overall_accuracy'] = player_stats['overall_accuracy'].clip(lower=0)
        player_stats = player_stats.sort_values(["tackle", "assist", "total"], ascending=False).reset_index(drop=True)
        return player_stats
    
    def calculate_team_wise_player_stats(self, overall_plays_with_tracking, season, team_filter):
        player_stats = self.calculate_player_stats()
        result = pd.merge(overall_plays_with_tracking, player_stats, on=["displayName"], how="inner")
        temp = result[["Season","Team", "displayName", "total", "tackle_y", "assist_y", "forced_fumble", "missed_tackle", "tackle_accuracy", 'assist_accuracy', 'forced_fumble_accuracy', 'missed_tackle_accuracy', 'overall_accuracy']]
        temp = temp[(temp['Season'] == season) & (temp['Team'] == team_filter)]
        grouped_data = temp.groupby(["Season","Team", "displayName"]).agg(
            total=pd.NamedAgg(column="total", aggfunc="first"),
            tackle=pd.NamedAgg(column="tackle_y", aggfunc="first"),
            assist=pd.NamedAgg(column="assist_y", aggfunc="first"),
            forced_fumble=pd.NamedAgg(column="forced_fumble", aggfunc="first"),
            missed_tackle=pd.NamedAgg(column="missed_tackle", aggfunc="first"),
            tackle_accuracy=pd.NamedAgg(column="tackle_accuracy", aggfunc="first"),
            assist_accuracy=pd.NamedAgg(column="assist_accuracy", aggfunc="first"),
            forced_fumble_accuracy=pd.NamedAgg(column="forced_fumble_accuracy", aggfunc="first"),
            missed_tackle_accuracy=pd.NamedAgg(column="missed_tackle_accuracy", aggfunc="first"),
            overall_accuracy=pd.NamedAgg(column="overall_accuracy", aggfunc="first")
        ).reset_index()

        grouped_data.rename(columns={'tackle_y': 'tackle', 'assist_y': 'assist'}, inplace=True)
        grouped_data_series = grouped_data.groupby(["Season","Team", "displayName", "total", "tackle", "assist", "forced_fumble", "missed_tackle", "tackle_accuracy", 'assist_accuracy', 'forced_fumble_accuracy', 'missed_tackle_accuracy', 'overall_accuracy'], as_index=True).size()
        team_wise_player_stats = pd.DataFrame(grouped_data_series, columns=['Count'])
        return team_wise_player_stats.drop(columns=['Count']).sort_values(["Team", "tackle", "assist", "overall_accuracy"], ascending=[True, False, False, False])
