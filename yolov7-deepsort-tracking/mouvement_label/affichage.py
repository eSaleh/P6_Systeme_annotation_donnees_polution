import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

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