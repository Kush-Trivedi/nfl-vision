import plotly.express as px
import plotly.graph_objects as go

class PlotlyGameVisualizer:
    def __init__(self, df):
        self.df = df
        
    def plot_game_in_plotly(self, gameId, playId):
        temp_tracking_df = self.df[(self.df['gameId'] == gameId) & (self.df['playId'] == playId)]
        for playId in temp_tracking_df['playId'].dropna().unique():
            temp_tracking_query = (self.df['gameId'] == gameId) & (self.df['playId'] == playId) 
            temp_tracking_df = (self.df[temp_tracking_query][['x', 'y','frameId', 'nflId', 'Team', 'displayName','playDescription','jerseyNumber','absoluteYardlineNumber','yardsToGo','down','playDirection','position','ballCarrierId','ballCarrierDisplayName']].fillna("").sort_values(['frameId']))
            playDescription = temp_tracking_df.playDescription.values[0]
            line_of_scrimmage = temp_tracking_df.absoluteYardlineNumber.values[0]
            yards_to_go = temp_tracking_df.yardsToGo.values[0]
            play_direction = temp_tracking_df.playDirection.values[0]
            down = temp_tracking_df.down.values[0]

            if play_direction == "left":
                first_down_marker = line_of_scrimmage - yards_to_go
            else:
                first_down_marker = line_of_scrimmage + yards_to_go

            fig = px.scatter(temp_tracking_df, x='x', y='y', animation_frame='frameId', color='Team',animation_group="nflId", hover_name="displayName",hover_data="displayName", custom_data=['jerseyNumber','displayName','position']) 
            fig.update_traces(marker=dict(size=16, line=dict(width=2, color='black')),selector=dict(mode='markers'), hovertemplate='%{customdata[0]} - %{customdata[1]} <br> %{customdata[2]}')

            for x in range(0, 20):
                for j in range(1, 5):
                    fig.add_trace(go.Scatter(x=[10 + x * 5 + j, 10 + x * 5 + j], y=[1, 3], mode='lines', line=dict(color='white', width=3),opacity=0.40, showlegend=False, hoverinfo='none'))
                    fig.add_trace(go.Scatter(x=[10 + x * 5 + j, 10 + x * 5 + j], y=[53.3 - 1, 53.3 - 3], mode='lines', line=dict(color='white', width=3),opacity=0.40, showlegend=False, hoverinfo='none'))


            y = (53.3 - 18.5) / 2
            for x in range(20):
                for j in range(1, 6):
                    if j == 5:
                        fig.add_trace(go.Scatter(x=[10 + x * 5 + j, 10 + x * 5 + j], y=[y, y + 2], mode='lines', line=dict(color='black', width=1),opacity=0.40, showlegend=False, hoverinfo='none'))
                        fig.add_trace(go.Scatter(x=[10 + x * 5 + j, 10 + x * 5 + j], y=[53.3 - y, 53.3 - y - 2], mode='lines', line=dict(color='black', width=1),opacity=0.40, showlegend=False, hoverinfo='none'))
                    else:
                        fig.add_trace(go.Scatter(x=[10 + x * 5 + j, 10 + x * 5 + j], y=[y, y + 2], mode='lines', line=dict(color='white', width=1),opacity=0.40, showlegend=False, hoverinfo='none'))
                        fig.add_trace(go.Scatter(x=[10 + x * 5 + j, 10 + x * 5 + j], y=[53.3 - y, 53.3 - y - 2], mode='lines', line=dict(color='white', width=1),opacity=0.40, showlegend=False, hoverinfo='none'))


            for x in range(0, 120, 10):
                fig.add_trace(go.Scatter(x=[x, x], y=[0, 53.3], mode='lines', showlegend=False,hoverinfo='none', line=dict(color="#333333"),opacity=0.50))

            for x in range(20, 110, 10):
                if x == 20 or x == 100: 
                    text_value = 10
                if x == 30 or x == 90: 
                    text_value = 20
                if x == 50: 
                    text_value = 40
                elif x < 50:
                    text_value = x - 10
                else:
                    text_value = 110 - x

                fig.add_trace(go.Scatter(x=[x, x], y=[10, 55], mode='text', text=str(text_value), showlegend=False, hoverinfo='none', textfont=dict(size=15, color='white'),opacity=0.40))
                fig.add_trace(go.Scatter(x=[x, x], y=[43.3, -10], mode='text', text=str(text_value), showlegend=False, hoverinfo='none', textfont=dict(size=15, color='white'),opacity=0.40))

            for i in range(3, 22): 
                fig.add_trace(go.Scatter(x=[5 * i, 5 * i], y=[1, 5], mode='lines', showlegend=False, hoverinfo='none', line=dict(color='white', width=3),opacity=0.40))
                fig.add_trace(go.Scatter(x=[5 * i, 5 * i], y=[53.3 - 1, 53.3 - 5], mode='lines', showlegend=False, hoverinfo='none', line=dict(color='white', width=3),opacity=0.40))

            fig.add_trace(go.Scatter(x=[0, 120], y=[53.3, 53.3], mode='lines', showlegend=False, hoverinfo='none', line=dict(color="#333333")))
            fig.add_trace(go.Scatter(x=[0, 120], y=[0, 0], mode='lines', showlegend=False, hoverinfo='none', line=dict(color="#333333")))
            fig.add_trace(go.Scatter(x=[line_of_scrimmage, line_of_scrimmage], y=[0, 53.3], mode='lines', line=dict(color='rgba(250, 160, 148, 1.00)', width=3, dash='dot'), showlegend=False, hoverinfo='none'))
            fig.add_trace(go.Scatter(x=[line_of_scrimmage], y=[-3], mode='text', text=f'Line of Scrimmage', showlegend=False, hoverinfo='none',textfont=dict(size=13, color='rgba(102, 157, 179, 1.00)')))
            fig.add_trace(go.Scatter(x=[first_down_marker, first_down_marker], y=[0, 53.3], mode='lines', line=dict(color='yellow', width=3, dash='dot'),opacity=0.80, showlegend=False, hoverinfo='none'))    
        
            fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 50
            fig.update_layout(autosize=False,width=1100,height=600,title=f'Game:{gameId}, Play: {playId} <br> {playDescription}',plot_bgcolor='white',xaxis=dict(showgrid=False,showticklabels=False),yaxis=dict(showgrid=False,showticklabels=False),yaxis_title = '',xaxis_title = '') 
            fig.update_layout(
                shapes=[
                    dict(type="rect",xref="x",yref="y",x0=0,y0=0,x1=120,y1=53.3,fillcolor="rgba(151, 188, 98, 1.00)",opacity=1,layer="below",line_width=0,),
                    dict(type="rect",xref="x",yref="y",x0=0,y0=0,x1=10,y1=53.3,fillcolor="black",opacity=1,layer="below",line_width=0),
                    dict(type="rect",xref="x",yref="y",x0=110,y0=0,x1=120,y1=53.3,fillcolor="black",opacity=1,layer="below",line_width=0)
                ],
                annotations=[
                    dict(text="HOME ENDZONE",x=5, y=26.65,font=dict(size=20, color='white'),showarrow=False,textangle=-90),
                    dict(text="VISITOR ENDZONE",x=115,y=26.65,font=dict(size=20, color='white'),showarrow=False,textangle=90),
                    dict(x=first_down_marker,y=53.0,text=str(down),showarrow=False,font=dict(family="Courier New, monospace",size=16,color="black"),
                    align="center",
                    bordercolor="black",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="#ff7f0e",
                    opacity=1)
                ]
            )
            fig.show()

