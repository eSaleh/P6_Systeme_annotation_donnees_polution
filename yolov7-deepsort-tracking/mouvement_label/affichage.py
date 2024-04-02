import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime, timedelta,timezone
from scipy.signal import find_peaks,argrelextrema

def objet_mouvement(data_json,frames_json):
    plt.figure(figsize=(15, len(data_json)*3))
    for j,object in enumerate(data_json) :
        frames = list(data_json[object].keys())
        x = np.array([data_json[object][frame][0] for frame in frames])
        y = np.array([data_json[object][frame][1] for frame in frames])
        v = np.array([data_json[object][frame][7] for frame in frames])
        a = np.array([data_json[object][frame][8] for frame in frames])
        t = [int(x) for x in frames]
        cmap = plt.jet

        frame_max = int(max(np.array(list(frames_json.keys())).astype(int)))

        plt.subplot (len(data_json),3,3*j+1)
        plt.plot(t,a)
        plt.grid()
        plt.title ('Object '+ object +' acceleration')
        plt.ylabel('Acceleration (pxl/frame²)')
        plt.xlabel('time (frame)')
        plt.xlim(0, frame_max)
        plt.ylim(-10, 10)

        plt.subplot (len(data_json),3,3*j+2)
        plt.plot(t,v)
        plt.grid()
        plt.title ('Object '+ object +' speed')
        plt.ylabel('Speed (pxl/frame)')
        plt.xlabel('time (frame)')
        plt.xlim(0, frame_max)
        plt.ylim(0, 20)

        plt.subplot (len(data_json),3,3*j+3)
        plt.scatter(x, 480 - y, c=t, cmap=cmap)
        plt.colorbar(label='Frame')
        plt.grid()
        plt.title ('Object '+ object +' position')
        plt.ylabel('y (pxl)')
        plt.xlabel('x (pxl)')
        plt.xlim(0, 640)
        plt.ylim(0, 480)
        plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.show()


def graph(fbf,date):
    time = list(fbf.keys())
    moving = [len(fbf[frame][0]) for frame in fbf]
    motionless = [len(fbf[frame][1]) for frame in fbf]

    # Create traces
    moving_trace = go.Scatter(x=time, y=moving, mode='lines', name='moving objects')
    motionless_trace = go.Scatter(x=time, y=motionless, mode='lines', name='motionless objects')

    # Create figure
    fig = go.Figure()
    
    # Add traces to figure
    fig.add_trace(moving_trace)
    fig.add_trace(motionless_trace)

    # Update layout
    fig.update_layout(title=f'Moving vs Motionless Objects over Time - {date}',
                      xaxis_title='Time',
                      yaxis_title='Number of Objects')

    # Show figure
    fig.show()

def graph_nb(nb,date):
    time = list(nb.keys())
    moving = [nb[frame] for frame in nb]
    #motionless = [len(nb[frame]) for frame in nb]

    # Create traces
    moving_trace = go.Scatter(x=time, y=moving, mode='lines', name='moving objects')
    #motionless_trace = go.Scatter(x=time, y=motionless, mode='lines', name='motionless objects')

    # Create figure
    fig = go.Figure()
    
    # Add traces to figure
    fig.add_trace(moving_trace)
    #fig.add_trace(motionless_trace)

    # Update layout
    fig.update_layout(title=f'Moving vs Motionless Objects over Time - {date}',
                      xaxis_title='Time',
                      yaxis_title='Number of Objects')

    # Show figure
    fig.show()

def graphe_capteurs(df,indicateur="pm2_5"):
    fig = go.Figure()
    for capteur in df["sensorId"].unique():
        capteur_name=str(capteur)
        if not (capteur_name.endswith("18") or capteur_name.endswith("75")): 
            fig.add_trace(go.Scatter(x=df[df["sensorId"]==capteur]['Date'], y=df[df["sensorId"]==capteur][indicateur], mode='lines',name=capteur_name))

    fig.update_layout(legend_title_text = f"{indicateur} vs time ")
    fig.update_xaxes(title_text="Time ")
    fig.update_yaxes(title_text=indicateur)
    fig.show()



def correlation_pm25(df_capteur_entreprise,local_maximum=False,width=10,year=2024,month=3,day=28,hour=11,minute=00,second=00):
    for capteur in df_capteur_entreprise["sensorId"].unique():
        if not (str(capteur).endswith("18") or str(capteur).endswith("75")):
            df_inter = df_capteur_entreprise[
                    (df_capteur_entreprise["sensorId"]==capteur) & 
                    (df_capteur_entreprise["Date"]>=datetime(year,month,day,hour,minute,second,tzinfo=timezone(timedelta(hours=1))))
                                    ].filter(regex='(passage|pm2_5)', axis=1)
            if local_maximum:
                df_inter["pm2_5"] = find_peaks(df_inter["pm2_5"],width)
                local_max_indexes=np.array(argrelextrema(df_inter.pm2_5.values, np.greater_equal,order=20)).flatten()
                local_max_indexes+=np.array([df_inter.index[0]]*len(local_max_indexes))

                # Optionally, get the indexes around the local maxima
                window_size = 5  # Adjust window size as needed
                indexes_around_maxima = []
                for index in local_max_indexes:
                    start_index = max(df_inter.index[0], index - window_size)
                    end_index = min(df_inter.index[-1], index + window_size + 1)
                    indexes_around_maxima.extend(range(start_index, end_index))

                # Remove duplicates and sort the indexes
                indexes_around_maxima = sorted(set(indexes_around_maxima))
                #print(local_max_indexes,df_inter.index[0])

                df_inter["pm2_5"].loc[df_inter.index.isin(indexes_around_maxima)]=1
                df_inter["pm2_5"].loc[~df_inter.index.isin(indexes_around_maxima)]=0
                #print(df_inter)

            plt.title(capteur)
            sns.heatmap(
                df_inter.corr(),annot=True)
            plt.show()