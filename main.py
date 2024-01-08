from data_visualization.nfl_stats import NFLStats
from data_preprocessesing.nfl_field import NFLFiled
from data_preprocessesing.nfl_data_loader import NFLDataLoader
from data_visualization.plotly_game_visualizer import PlotlyGameVisualizer
from data_visualization.matplotlib_game_visualizer import MatplotlibGameVisualizer


# Create NFL Field
field = NFLFiled(120, 53.3)
field.save_pitch('../nfl-vision/assets','pitch.png')

# Load Required Data
nfl_data_loader = NFLDataLoader()
games_df = nfl_data_loader.load_data("games.csv")
players_df = nfl_data_loader.load_data("players.csv")
tackles_df = nfl_data_loader.load_data("tackles.csv")
nfl_data_loader.load_all_data()
overall_plays_with_tracking = nfl_data_loader.get_overall_plays_with_tracking()

# Visualize Players Stats
players_stats = NFLStats(games_df, players_df, tackles_df)
overall_player_stats = players_stats.calculate_player_stats()
overall_player_stats = overall_player_stats.drop(columns=["Season"])
team_wise_player_stats_per_season = players_stats.calculate_team_wise_player_stats(overall_plays_with_tracking,2022,"KC")

# Visualize Game Play of your choice
plotly_visualizer = PlotlyGameVisualizer(overall_plays_with_tracking)
matplotlib_visualizer = MatplotlibGameVisualizer(overall_plays_with_tracking)
