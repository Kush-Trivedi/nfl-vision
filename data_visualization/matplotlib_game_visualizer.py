# Import Libraries
import io
import os
import copy
import time
import boto3
import pickle
import shutil
import imageio
import traceback
import numpy as np
import pandas as pd
from PIL import Image
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from moviepy.editor import ImageSequenceClip
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle, Ellipse, Patch

import matplotlib
matplotlib.use('Agg')

# Constant Graphics Params
plt.rcParams['figure.dpi'] = 180
plt.rcParams["figure.figsize"] = (22,16)
sns.set(rc={
    'axes.facecolor':'#FFFFFF', 
    'figure.facecolor':'#FFFFFF',
    'font.sans-serif':'DejaVu Sans',
    'font.family':'sans-serif'
})

# Custom Legend Handler Class
class HandlerColormap(HandlerBase):
    def __init__(self, cmap, num_rectangles=8, **kw):
        HandlerBase.__init__(self, **kw)
        self.cmap = cmap
        self.num_rectangles = num_rectangles
    
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        rectangles = []
        segment_width = width / self.num_rectangles
        
        for i in range(self.num_rectangles):
            r = Rectangle([xdescent + i * segment_width, ydescent], segment_width, height, fc=self.cmap(i / self.num_rectangles),transform=trans)
            rectangles.append(r)
        return rectangles

class MatplotlibGameVisualizer:
    def __init__(self, df):
        self.df = df
        self.pitch_img = Image.open('../assets/pitch.png')
        self.s3_client = boto3.client('s3')
        
    def calculate_distance(self,x1, y1, x2, y2):
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

    def angle_with_x_axis(self, x, y):
        return np.degrees(np.arctan2(y, x))

    def angle_in_32_segments(self, angle):
        if np.isnan(angle):
            return 0 
        angle = angle % 360
        return round(angle / 11.25)
    
    def calculate_tackling_probability(self, distance, player_direction, player_orientation, player_speed, player_acceleration, player_x, player_y, ball_x, ball_y, ball_speed, ball_acceleration, ball_direction):
        # Normalization constants for the defensive player
        max_player_speed_norm = 3.95  # yards/second
        max_player_acceleration_norm = 2.46  # yards/second²

        # Normalization constants for the ball carrier (opposing player)
        max_ball_speed_norm = 5.88  # Example value, adjust based on data
        max_ball_acceleration_norm = 4.19  # Example value, adjust based on data

        # Calculate angles
        direction_to_ball = np.degrees(np.arctan2(ball_y - player_y, ball_x - player_x))
        angle_difference_direction = abs(player_direction - direction_to_ball)
        angle_difference_ball = abs(ball_direction - direction_to_ball)

        # Convert angles to 32 segment scale
        segment_difference_direction = self.angle_in_32_segments(angle_difference_direction)
        segment_difference_ball = self.angle_in_32_segments(angle_difference_ball)

        # Movement checks
        moving_towards_ball = segment_difference_direction < (90 / 11.25)
        ball_moving_away = segment_difference_ball > (90 / 11.25)

        # Speed and acceleration factors
        player_speed_factor = min(player_speed / max_player_speed_norm, 1)
        player_acceleration_factor = min(player_acceleration / max_player_acceleration_norm, 1)
        ball_speed_factor = min(ball_speed / max_ball_speed_norm, 1)
        ball_acceleration_factor = min(ball_acceleration / max_ball_acceleration_norm, 1)

        # Base probability calculation
        if distance >= 7:
            base_probability = 0
        elif 5 <= distance < 7:
            base_probability = (1 / (distance + 1)) / (distance / 2)
        else:
            base_probability = 1 / (distance + 1)

        # Adjust base probability based on player's state
        if moving_towards_ball:
            base_probability *= (1 + 0.75 * player_speed_factor + 0.75 * player_acceleration_factor)
        else:
            base_probability *= (1 + 0.25 * player_speed_factor + 0.25 * player_acceleration_factor)

        # Reduce probability if ball is moving away
        # Consider both speed and acceleration of the ball
        if ball_moving_away:
            base_probability *= (1 - ball_speed_factor * ball_acceleration_factor)

        return min(base_probability, 1)
        
    def calculate_player_density(self,node, radius, G, pos):
        player_count = 0
        for u, v, d in G.edges(data=True):
            if u == node:
                distance = ((pos[u][0] - pos[v][0])**2 + (pos[u][1] - pos[v][1])**2)**0.5
                if distance <= radius:
                    player_count += 1
            elif v == node:
                distance = ((pos[u][0] - pos[v][0])**2 + (pos[u][1] - pos[v][1])**2)**0.5
                if distance <= radius:
                    player_count += 1
        return player_count
    
    def calculate_dx_dy_arrow(self, x, y, angle, speed, multiplier):
        fixed_length = 0.5
        angle_radians = np.radians(angle)
        
        if angle <= 90:
            dx = np.sin(angle_radians) * fixed_length
            dy = np.cos(angle_radians) * fixed_length
        elif angle <= 180:
            angle_radians = np.radians(angle - 90)
            dx = np.sin(angle_radians) * fixed_length
            dy = -np.cos(angle_radians) * fixed_length
        elif angle <= 270:
            angle_radians = np.radians(angle - 180)
            dx = -np.sin(angle_radians) * fixed_length
            dy = -np.cos(angle_radians) * fixed_length
        else: 
            angle_radians = np.radians(360 - angle)
            dx = -np.sin(angle_radians) * fixed_length
            dy = np.cos(angle_radians) * fixed_length
        return dx, dy
    
    def assign_direction(self, angle):
        # Define the 32 compass points and their corresponding bounds
        directions = [
            "North", "North by East", "North-Northeast", "Northeast by North",
            "Northeast", "Northeast by East", "East-Northeast", "East by North",
            "East", "East by South", "East-Southeast", "Southeast by East",
            "Southeast", "Southeast by South", "South-Southeast", "South by East",
            "South", "South by West", "South-Southwest", "Southwest by South",
            "Southwest", "Southwest by West", "West-Southwest", "West by South",
            "West", "West by North", "West-Northwest", "Northwest by West",
            "Northwest", "Northwest by North", "North-Northwest", "North by West",
            "North"  
        ]
        
        bounds = [i * 11.25 for i in range(33)]

        for i in range(len(bounds) - 1):
            lower_bound = bounds[i]
            upper_bound = bounds[i + 1]
            if lower_bound <= angle < upper_bound:
                return directions[i]

        return None

    def calculate_relative_velocity(self, speed1, speed2, dir1, dir2):
        # Convert directions to radians
        theta1 = np.radians(dir1)
        theta2 = np.radians(dir2)

        # Calculate velocity components
        vx1 = speed1 * np.cos(theta1)
        vy1 = speed1 * np.sin(theta1)
        vx2 = speed2 * np.cos(theta2)
        vy2 = speed2 * np.sin(theta2)

        # Calculate relative velocity components
        rel_vx = vx1 - vx2
        rel_vy = vy1 - vy2

        # Calculate magnitude of relative velocity
        return np.sqrt(rel_vx**2 + rel_vy**2)

    def calculate_time_to_contact(self, distance, rel_velocity):
        return distance / rel_velocity if rel_velocity > 0 else np.inf

    def calculate_angle_of_approach(self, dir1, dir2):
        # Convert directions to radians and calculate vectors
        vector1 = np.array([np.cos(np.radians(dir1)), np.sin(np.radians(dir1))])
        vector2 = np.array([np.cos(np.radians(dir2)), np.sin(np.radians(dir2))])

        # Calculate angle in radians
        dot_product = np.dot(vector1, vector2)
        angle = np.arccos(dot_product / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))

        # Convert angle to degrees
        return np.degrees(angle)

    def create_s3_bucket(self, bucket_name):
        try:
            self.s3_client.create_bucket(Bucket=bucket_name)
            print(f"Bucket {bucket_name} created.")
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                print(f"Bucket {bucket_name} already exists and owned by you.")
            else:
                print(f"Error in creating bucket: {e}")

    def upload_to_s3(self, bucket_name, buffer, object_name, content_type=None):
        try:
            extra_args = {'ContentType': content_type} if content_type else None
            self.s3_client.upload_fileobj(buffer, bucket_name, object_name, ExtraArgs=extra_args)
        except NoCredentialsError:
            print("Credentials not available")
        except Exception as e:
            print(f"Error during upload: {e}")


    def resize_for_video(self, image, target_size):
        return image.resize(target_size, Image.LANCZOS)

    def process_frames(self, frames):
        # Determine a consistent target size divisible by 16 for all frames
        # Example: Use the size of the first frame or define a specific size
        first_width, first_height = frames[0].size
        target_width = (first_width + 15) // 16 * 16
        target_height = (first_height + 15) // 16 * 16
        target_size = (target_width, target_height)

        # Resize all frames to the target size
        resized_frames = [self.resize_for_video(frame, target_size) for frame in frames]
        return resized_frames


    def plot_game_in_matplotlib(self, gameId, playId):
        # Validate the Result
        if not ((self.df['gameId'] == gameId) & (self.df['playId'] == playId)).any():
            raise ValueError("No data available for the provided gameId and playId.")

        # Define S3 Bucket Name
        bucket_name = 'nfl-big-data-bowl-2024'

        # Create S3 Bucket if it does not exist
        self.create_s3_bucket(bucket_name)

        # Check if the GIF already exists in S3
        gif_object_name = f"game_plays/{gameId}_{playId}/animation.gif"
        try:
            self.s3_client.head_object(Bucket=bucket_name, Key=gif_object_name)
            print("GIF already exists in S3. Skipping upload.")
            return
        except:
            print("GIF does not exist in S3. Proceeding with creation and upload.")
           
        # Extract Necessary Data and Manipulate
        game_play_df = self.df[(self.df['gameId'] == gameId) & (self.df['playId'] == playId)]
        defensive_team = game_play_df.defensiveTeam.values[0]
        possession_team = game_play_df.possessionTeam.values[0]
        unique_frame_ids = game_play_df['frameId'].unique()
        play_description = game_play_df.playDescription.values[0]
        offense_formation = game_play_df.offenseFormation.values[0]
        line_of_scrimmage = game_play_df.absoluteYardlineNumber.values[0]
        yard_line_number = game_play_df.yardlineNumber.values[0]
        down = game_play_df.down.values[0]
        quarter = game_play_df.quarter.values[0]
        play_direction = game_play_df.playDirection.values[0]
        yards_to_go = game_play_df.yardsToGo.values[0]
        defenders_in_the_box = game_play_df.defendersInTheBox.values[0]
        pass_probability = game_play_df.passProbability.values[0]
        expected_points = game_play_df.expectedPoints.values[0]
        expected_points_added = game_play_df.expectedPointsAdded.values[0]
        ballCarrierName = game_play_df.ballCarrierDisplayName.values[0]
        playResult = game_play_df.playResult.values[0]
        unique_events = game_play_df.event.unique()
        unique_events_list = list(unique_events)
        events_str = ', '.join(map(str, unique_events_list))

        down_distance_efficiency = playResult / (down * yards_to_go)
        yard_line_advantage = (100 - yard_line_number) * expected_points_added

        featureList = []
        featureList.append(down)
        featureList.append(yards_to_go)
        featureList.append(yard_line_number)
        featureList.append(defenders_in_the_box)
        featureList.append(pass_probability)
        featureList.append(expected_points)
        featureList.append(expected_points_added)
        featureList.append(down_distance_efficiency)
        featureList.append(yard_line_advantage)

        columnNames = [
            'down',
            'yardsToGo',
            'yardlineNumber',
            'defendersInTheBox',
            'passProbability',
            'expectedPoints',
            'expectedPointsAdded',
            'DownDistanceEfficiency',
            'YardlineAdvantage'
        ]

        feature_df = pd.DataFrame([featureList], columns=columnNames)
        model = pickle.load(open("/Users/kushtrivedi/Desktop/NFL-Big-Data-Bowl-2024/models/stacking_model.pkl", 'rb'))
        model_output = model.predict(feature_df)
        expected_yard = model_output.round().astype(int)
        expected_yard_result = expected_yard[0]
        
        if play_direction == "left":
            first_down_marker = line_of_scrimmage - yards_to_go
            if expected_yard_result >= -1:
                expected_result = line_of_scrimmage - expected_yard_result
            else:
                expected_result = line_of_scrimmage + abs(expected_yard_result)
        else:
            first_down_marker = line_of_scrimmage + yards_to_go
            if expected_yard_result >= -1:
                expected_result = line_of_scrimmage + expected_yard_result
            else:
                expected_result = line_of_scrimmage - abs(expected_yard_result)
        
        # Extract Frame by Frame
        frames = []
        extra_space = "\n\n"
        for frameId in unique_frame_ids:
            # Get Ball, Defense and Offense Data
            frame_info_str = "Short Summary:" + extra_space
            frame_data = game_play_df[game_play_df['frameId'] == frameId]
            ball_carrier_data = frame_data[frame_data['Team'] == 'football']
            defensive_players_data = frame_data[frame_data['Team'] == defensive_team]
            extract_defense_formation = defensive_players_data['position'].value_counts()
            defense_formation = ', '.join([f"{count}- {position}" for position, count in extract_defense_formation.items()])
            offense_player_data = frame_data[frame_data['Team'] == possession_team]
            events = frame_data['event'].astype(str).unique()
            event_name = ', '.join(events)

            # Short summary of the game play
            initial_play_details = f"- Play Description: Game Clock{play_description}\n- Quarter: {quarter}, Line of Scrimmage at {line_of_scrimmage} yards, Down: {down}, Yards To Go: {yards_to_go} yards, Expected Play Result from Line of Scrimmage: {expected_yard} yards\n\n- Defense Team:{defensive_team} and is playing with {defense_formation}\n- Offense Team: {possession_team} is playing {offense_formation} formation where Ball Carrier his {ballCarrierName} and play result was {playResult} yards\n\n"
            frame_info_str += initial_play_details
            defense_details = "Defense Details:\n"
            frame_info_str += defense_details
            for index, row in defensive_players_data.iterrows():
                def_player_str = f"- Defense Player ({row['jerseyNumber']}) {row['displayName']} is playing position {row['position']} and has a weight of {row['weight']}lbs and height is {row['height']} inch\n"
                frame_info_str += def_player_str
                
            offense_details = extra_space + "Offesne Details:\n"
            frame_info_str += offense_details
            for index, row in offense_player_data.iterrows():
                off_player_str = f"- Offense Player ({row['jerseyNumber']}) {row['displayName']} is playing position {row['position']} and has a weight of {row['weight']}lbs and height is {row['height']} inch\n"
                frame_info_str += off_player_str
    
            # Get Ball and Player Data to plot zoomed plot
            ball_x = ball_carrier_data['x'].values[0]
            ball_y = ball_carrier_data['y'].values[0]
            offense_x = offense_player_data['x']
            offense_y = offense_player_data['y']
            defensive_x = defensive_players_data['x']
            defensive_y = defensive_players_data['y']
            
            # Calculate min and max coordinates for x and y axes
            min_x = min(min(offense_x), min(defensive_x), ball_x) - 5
            max_x = max(max(offense_x), max(defensive_x), ball_x) + 5
            min_y = min(min(offense_y), min(defensive_y), ball_y) - 5
            max_y = max(max(offense_y), max(defensive_y), ball_y) + 5 
            
            # Initialize Fig Object
            fig, ax = plt.subplots()
            ax.imshow(self.pitch_img, extent=[0, 120, 0, 53.3],aspect='auto') 
            ax.axvline(x=line_of_scrimmage, color='#00539CFF', linestyle='-', linewidth=4)
            ax.axvline(x=first_down_marker, color='#FDD20EFF', linestyle='-', linewidth=4)
            ax.axvline(x=expected_result, color='#7851A9', linestyle='-', linewidth=4)
        
            # Add Section 1
            frame_str = extra_space + f"Frame {frameId}:\n\n- Event happening: {event_name}\n\n"
            frame_info_str += frame_str
            section_1 = "Defensive Positioning and Tackling Likelihood: Proximity to Football with Probability Categorization; Probability Ranges: High Tackle Likelihood (90-100%), Potential Assists or Misses (61-80%), Approaching Players (41-60%)\n"
            frame_info_str += section_1

            # Defensive player plotting
            for _, player in defensive_players_data.iterrows():
                x1, y1 = player['x'], player['y'] 
                x2, y2 = ball_x, ball_y 
                distance = self.calculate_distance(x1,y1,x2,y2)
                player_direction = player['dir'] 
                player_speed = player['s']  
                player_acceleration = player['a']
                player_orientation = player['o']
                ball_speed = ball_carrier_data['s'].values[0]
                ball_direction = ball_carrier_data['dir'].values[0]  
                ball_acceleration = ball_carrier_data['a'].values[0]
                probability = self.calculate_tackling_probability(distance, player_direction, player_orientation, player_speed, player_acceleration, x1, y1, x2, y2, ball_speed,ball_acceleration, ball_direction)
                jersey_number = int(player['jerseyNumber'])
                player_info = f"* Defense Player: ({jersey_number})-{player['displayName']}, Distance from Ball is {distance:.2f} yards, Pressure Tackling Probability is {probability*100:.2f}%\n"
                frame_info_str += player_info
                
                if 41 <= int(probability*100) <= 101:
                    radius = 0.8
                    num_points = 100
                    theta = np.linspace(0, 2*np.pi, num_points)
                    r = np.linspace(0, radius, num_points)
                    R, Theta = np.meshgrid(r, theta)
                    X, Y = R*np.cos(Theta) + x1, R*np.sin(Theta) + y1
                    Z = np.ones_like(X) * probability
                    Z = Z * np.exp(-0.5 * (R / radius)**2)

                    prob_int = int(probability * 100)
                    
                    if 91 <= prob_int <= 98:
                        cmap = plt.get_cmap('Reds')
                    elif 61 <= prob_int <= 80:
                        cmap = plt.get_cmap('YlGn')
                    elif 41 <= prob_int <= 60:
                        cmap = plt.get_cmap('Blues')
                   
                    norm = mcolors.Normalize(vmin=Z.min(), vmax=Z.max())
                    ax.contourf(X, Y, Z, cmap=cmap, alpha=0.7, norm=norm, levels=100)
                    
                if 5 < distance < 7:
                    ax.plot([x1, x2], [y1, y2], color='#FF3EA5FF', linestyle='--', linewidth=2, alpha=0.6)
                elif distance < 5:
                    ax.plot([x1, x2], [y1, y2], color='red', linestyle='-', linewidth=2, alpha=0.9)

                ax.text(player['x'], player['y'], str(jersey_number), color='white', ha='center', va='center', fontsize=12, weight='bold')
                
                position_part = f'{player["position"].ljust(2)}: '
                jersey_part = f'{int(player["jerseyNumber"])+0:02}'
                name_part = f'{player["displayName"].ljust(20)}'
                tackle_probability = f'Pressure: [{probability*100:02.0f}%]  '
                
                label = tackle_probability + position_part + jersey_part + ' - ' + name_part
                ax.scatter(x1, y1, color="#964F4CFF", s=500, ec='k', label=label)
                ax.margins(0.1)
            
            # Add Section 2
            frame_info_str += extra_space 
            section_2 = "Defensive Spatial Analysis: Teammates Distances for Defense Tactical Insights\n"
            frame_info_str += section_2 

            # Graph Ploting for Defense
            G = nx.Graph() 
            
            for _, row in defensive_players_data.iterrows():
                G.add_node(row['displayName'], pos=(row['x'], row['y']))

            for node1 in G.nodes:
                for node2 in G.nodes:
                    if node1 != node2:
                        x1, y1 = G.nodes[node1]['pos']
                        x2, y2 = G.nodes[node2]['pos']
                        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                        if distance < 5:
                            G.add_edge(node1, node2, weight=1/distance)

            pos = nx.get_node_attributes(G, 'pos')
            nx.draw_networkx(G, pos, with_labels=False, node_size=500, font_size=12, font_weight='bold', node_color='#964F4CFF')
            cycles = list(nx.simple_cycles(G))
            nx.draw_networkx_nodes(G, pos, node_size=500, node_color='#964F4CFF',alpha=0.2)
            nx.draw_networkx_edges(G, pos, edgelist=cycles, edge_color='black', width=2, alpha=0.4)

            # Plot Polygon for Defense
            for cycle in cycles:
                cycle_nodes = cycle + [cycle[0]]
                cycle_pos = np.array([pos[node] for node in cycle_nodes])
                polygon = plt.Polygon(cycle_pos, closed=True, fill=True, color='#6DAC4FFF', alpha=0.1)
                plt.gca().add_patch(polygon)
       
            for u, v, d in G.edges(data=True):
                is_in_cycle = any(u in cycle and v in cycle for cycle in cycles)

                if is_in_cycle:
                    continue
     
            for u, v, d in G.edges(data=True):
                ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], color='black', linestyle='-', linewidth=2, alpha=0.4)
                distance = 1 / d['weight']
                player_distance_info = f"* Distance between Defense Player {u} and Defense Player {v} is {distance:.2f} yards\n"
                frame_info_str += player_distance_info
                center_x = 0.5 * (pos[u][0] + pos[v][0])
                center_y = 0.5 * (pos[u][1] + pos[v][1])
                ax.annotate(f'{distance:.0f} yd', xy=(center_x, center_y), ha='center', va='bottom', fontsize=13, weight='bold', color='#4F3466FF', alpha=1)
    
            frame_info_str += extra_space

            # Plot arrows for Defensive players
            close_proximity_color = 'red'
            ax = plt.gca()

            for _, def_row in defensive_players_data.iterrows():
                defense_start_x = def_row['x']
                defense_start_y = def_row['y']
                defense_team = def_row['Team'] 
                defense_color = 'black' 

                for _, off_row in offense_player_data.iterrows():
                    offense_start_x = off_row['x']
                    offense_start_y = off_row['y']
                    offense_team = off_row['Team'] 

                    distance = self.calculate_distance(defense_start_x, defense_start_y, offense_start_x, offense_start_y)

                    if distance <= 1 and defense_team != offense_team:
                        defense_color = close_proximity_color
                        break 

                dx_dir, dy_dir = self.calculate_dx_dy_arrow(defense_start_x, defense_start_y, def_row['dir'], def_row['s'], 1)
                ax.arrow(defense_start_x, defense_start_y, dx_dir, dy_dir, color=defense_color,ec='black', width=0.25, head_width=0.5, head_length=0.2, shape='full', alpha=0.7)
            
            # Add Section 3
            section_3 = "Offensive Players Spatial Analysis: Proximity to Nearest Defensive Players for Offense Tactical Insights and If Distance less than 1 yard offesne player is likely to have a contact with defense player\n"
            frame_info_str += section_3

            # Plot arrows for Offensive players
            for _, off_row in offense_player_data.iterrows():
                offense_start_x = off_row['x']
                offense_start_y = off_row['y']
                offense_jersey_number = int(off_row['jerseyNumber'])
                offense_team = off_row['Team']
                offense_color = 'black'
                
                position_part = f'{off_row["position"].ljust(3)}: '
                jersey_part = f'{int(off_row["jerseyNumber"])+0:02}'
                name_part = f'{off_row["displayName"].ljust(20)}'
                label = position_part + jersey_part + ' - ' + name_part
                ax.scatter(offense_start_x, offense_start_y, color='black', s=500, label=label, alpha=0.6)
                ax.text(offense_start_x, offense_start_y, str(int(offense_jersey_number)), color='white', ha='center', va='center', fontsize=12, weight='bold')

                # Get movement and looking direction for the offensive player
                off_player_moving = self.assign_direction(off_row['dir'])
                off_player_looking = self.assign_direction(off_row['o'])
                off_speed = off_row['s'] 
                off_acceleration = off_row['a']
                off_mass_lbs = off_row['weight'] 
                off_momentum = off_mass_lbs * off_speed
                off_force = round(off_mass_lbs * off_acceleration, 2)
                off_kinetic_energy = 0.5 * off_mass_lbs * (off_speed ** 2)
                closest_def_players = []

                for _, def_row in defensive_players_data.iterrows():
                    defense_start_x = def_row['x']
                    defense_start_y = def_row['y']
                    defense_team = def_row['Team']
                    distance = self.calculate_distance(offense_start_x, offense_start_y, defense_start_x, defense_start_y)
                    closest_def_players.append((distance, def_row))
                    

                    if distance <= 1 and offense_team != defense_team:
                        offense_color = close_proximity_color
                        break 

                # Sort by distance and select top 3
                closest_def_players.sort(key=lambda x: x[0])
                top_3_closest_def = closest_def_players[:3]
                frame_info_str += f"* Offense Player ({offense_jersey_number})-{off_row['displayName']}: Moving {off_player_moving} ({off_row['dir']:.2f}°) - Looking {off_player_looking} ({off_row['o']:.2f}°) and Running at at a Speed of {off_row['s']:.2f} yards/second, Momentum of {off_momentum:.2f} lb-yd/s^2, Applying Force of {off_force:.2f} lb-yd/s^2, Kinetic energy of {off_kinetic_energy:.2f} lb-yd^2/s^2. Closest Defense Players: "

                for dist, def_player in top_3_closest_def:
                    defense_jersey_number = int(def_player['jerseyNumber'])
                    def_player_moving = self.assign_direction(def_player['dir'])
                    def_player_looking = self.assign_direction(def_player['o'])
                    def_speed = def_player['s'] 
                    def_acceleration = def_player['a']
                    def_mass_lbs = def_player['weight'] 
                    def_momentum = def_mass_lbs * def_speed
                    def_force = round(def_mass_lbs * def_acceleration, 2)
                    def_kinetic_energy = 0.5 * def_mass_lbs * (def_speed ** 2)
                    # Calculate relative velocity, time to contact, and angle of approach (Offense to Defense)
                    rel_velocity_off_def = self.calculate_relative_velocity(off_speed, def_speed, off_row['dir'], def_player['dir'])
                    time_to_contact_off_def = self.calculate_time_to_contact(dist, rel_velocity_off_def)
                    angle_of_approach_off_def = self.calculate_angle_of_approach(off_row['dir'], def_player['dir'])
                    # Calculate relative velocity, time to contact, and angle of approach (Defense to Offense)
                    rel_velocity_def_off = self.calculate_relative_velocity(def_speed, off_speed, def_player['dir'], off_row['dir'])
                    time_to_contact_def_off = self.calculate_time_to_contact(dist, rel_velocity_def_off)
                    angle_of_approach_def_off = self.calculate_angle_of_approach(def_player['dir'], off_row['dir'])

                    frame_info_str += f"({defense_jersey_number})-{def_player['displayName']}: {dist:.2f} yds, Moving {def_player_moving} ({def_player['dir']:.2f}°) - Looking {def_player_looking} ({def_player['o']:.2f}°), and Running at a Speed of {def_speed:.2f} yards/second, Momentum of {def_momentum:.2f} lb-yd/s^2, Applying Force of {def_force:.2f} lb-yd/s^2, Kinetic energy of {def_kinetic_energy:.2f} lb-yd^2/s^2, Relative Velocity (Off-Def): {rel_velocity_off_def:.2f} yd/s, Time to Contact (Off-Def): {time_to_contact_off_def:.2f} sec, Angle of Approach (Off-Def): {angle_of_approach_off_def:.2f}°, Relative Velocity (Def-Off): {rel_velocity_def_off:.2f} yd/s, Time to Contact (Def-Off): {time_to_contact_def_off:.2f} sec, Angle of Approach (Def-Off): {angle_of_approach_def_off:.2f}°; "

                frame_info_str = frame_info_str.rstrip('; ') + extra_space
                        
                dx_dir, dy_dir = self.calculate_dx_dy_arrow(offense_start_x, offense_start_y, off_row['dir'], off_row['s'], 1)
                ax.arrow(offense_start_x, offense_start_y, dx_dir, dy_dir, color=offense_color,ec='black', width=0.25, head_width=0.5, head_length=0.2, shape='full', alpha=0.7)
            
            # Added 3 Sections
            frame_info_str += extra_space            
           
            # Upload Text File
            text_buffer = io.StringIO()
            text_buffer.write(frame_info_str)
            text_buffer.seek(0)
            bytes_buffer = io.BytesIO(text_buffer.getvalue().encode())
            text_object_name = f"game_plays/{gameId}_{playId}/{gameId}_{playId}_{frameId:04d}.txt"
            self.upload_to_s3(bucket_name, bytes_buffer, text_object_name, content_type='text/plain')
            text_buffer.close()
            bytes_buffer.close()

            # Plot Football
            ax.add_artist(Ellipse((ball_x, ball_y), 0.55, 0.5, facecolor="#755139FF", ec="#F2EDD7FF", lw=2))
    
            # Adjust Plot Design
            ax.axes.get_yaxis().set_visible(False)
            ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)
            ax.set_axis_on()
            ax.axes.get_xaxis().set_visible(True)
            ax.set_xlabel(None)
            
            # Plot Legends and Text
            words = play_description.split()
            formatted_play_description = '\n'.join(' '.join(words[i:i+15]) for i in range(0, len(words), 17))
            title_str = f'FRAME: {frameId}      ' + '✤ Play Description: ' + formatted_play_description + "\n✤ Offense Formation: " + offense_formation + "     " + "✤ Defense Formation: " + defense_formation
            ax.set_title(title_str, x=0.5, y=1, fontweight='bold', fontsize=18)
            
            # Top Legend
            top_handles = [
                Line2D([0], [0], marker='o', color='w', label= defensive_team, markersize=20, markerfacecolor='#964F4CFF',markeredgecolor='k'),
                Line2D([0], [0], marker='o', color='w', label=possession_team, markersize=20, markerfacecolor='gray',markeredgecolor='k'),
                Line2D([0], [0], marker='|', color='#00539CFF', label=line_of_scrimmage-10, linestyle='None',markersize=20, markeredgewidth=4),
                Line2D([0], [0], marker='|', color='#FDD20EFF', label=down, linestyle='None',markersize=20, markeredgewidth=4)
            ]

            top_labels = [f'Defense: {defensive_team}', f'Offense : {possession_team}',f' LOS : {line_of_scrimmage-10} yd', f'Down: {down}']
            top_legend = ax.legend(title="Team and Game Situation", handles=top_handles, labels=top_labels, loc='center left', bbox_to_anchor=(1, 0.95), fontsize='x-large',ncol=2,title_fontsize=22)
            ax.add_artist(top_legend)
            
            # Plot Cmap Legend
            cmap_patch = Patch(color='#6DAC4FFF',ec='black', label='Strong Defense')
            cmaps = [plt.get_cmap("Reds"), plt.get_cmap("YlGn"), plt.get_cmap("Blues")]
            cmap_handles = [Rectangle((0, 0), 1, 1) for _ in cmaps] + [cmap_patch]
            cmap_labels = ["Defense Player May Tackle", "Defense Player May Assist", "Defense Player is Closer", "Strong Defense Area"]
            line_handle = Line2D([0], [0], color='#7851A9', label=f'Expected Play Result {expected_yard_result} yd', marker='|',linestyle='None',markersize=20, markeredgewidth=4)
            cmap_handles.append(line_handle)
            cmap_labels.append(f'Expected Play Result {expected_yard_result} yd') 
            handler_map = dict(zip(cmap_handles,[HandlerColormap(cm) for cm in cmaps]))
            cmap_legend = ax.legend(handles=cmap_handles,labels=cmap_labels, handler_map=handler_map,loc='upper center', bbox_to_anchor=(0.66, -0.01), fontsize='xx-large', ncol=5)
            ax.add_artist(cmap_legend)
            
            # Plot Arrow Legend
            arrow_handles = [
                Line2D([0],[0],label="Gap Between Defense Players in Yards",color='black',linewidth=3),
                Line2D([0], [0], label='Has Good Chance of Tackle', color='red', linewidth=3),
                Line2D([0], [0], label='Has Low Chance of Tackle', color='#FF3EA5FF', linestyle="--", linewidth=3),
                Line2D([0], [0], label='Defense Player in Close Proximity to Offense ', marker='>', markersize=20, markeredgecolor='black', markerfacecolor='red', linestyle='-', color="black", linewidth=3),
                Line2D([0], [0], label='Defense/Offense Players Moving Direction', marker='>', markersize=20, markeredgecolor='black', markerfacecolor='black', linestyle='-', color="black", linewidth=3)
            ]
            arrow_legend = ax.legend(title="Players Proximity and Movement",handles=arrow_handles, loc='center left', bbox_to_anchor=(1, 0.09), fontsize='x-large',title_fontsize=22)
            ax.add_artist(arrow_legend)
            
            # Plot Player Legend
            ax.legend(title="Players Roster",loc='center left', bbox_to_anchor=(1, 0.542), fontsize='x-large',title_fontsize=22)
                 
            # Set the limits for x and y axes
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.axis('off')

            # Convert the plt figure (matplotlib figure object) to an image
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', bbox_inches='tight')
            img_buffer.seek(0)
            img = Image.open(img_buffer)
            frames.append(img.copy()) 
            img_buffer.seek(0)  
    
            # Upload each frame to S3
            frame_object_name = f"game_plays/{gameId}_{playId}/{gameId}_{playId}_{frameId:04d}.png"
            self.upload_to_s3(bucket_name, img_buffer, frame_object_name, content_type='image/png')
            img_buffer.close() 

            plt.cla() 
            ax.clear()
            plt.close(fig)
 
        # Create and Upload GIF to S3
        gif_buffer = io.BytesIO()
        frames[0].save(gif_buffer, format='GIF', save_all=True, append_images=frames[1:], loop=0, duration=120)
        gif_buffer.seek(0) 
        gif_object_name = f"game_plays/{gameId}_{playId}/animation.gif"
        self.upload_to_s3(bucket_name, gif_buffer, gif_object_name, content_type='image/gif')
        gif_buffer.close()
    
        # Create MP4
        video_buffer = io.BytesIO()
        processed_frames = self.process_frames(frames)
        with imageio.get_writer(video_buffer, format='mp4', fps=6) as writer:
            for resized_frame in processed_frames:
                writer.append_data(np.array(resized_frame))
        
        video_buffer.seek(0)
        video_object_name = f"game_plays/{gameId}_{playId}/animation.mp4"
        self.upload_to_s3(bucket_name, video_buffer, video_object_name, content_type='video/mp4')
        video_buffer.close()
        time.sleep(10)
     
