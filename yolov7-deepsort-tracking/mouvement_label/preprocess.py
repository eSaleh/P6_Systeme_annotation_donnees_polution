import os 
import json
import re 
import numpy as np 

def read_jsons(PATH,num_cam="",date=""):
    """    Renvoi les json obtenu à partir de l'algorithme de détection et de suivi d'objets


    Args:
        PATH (_type_): Chemin d'accès des jsons à lire
        num_cam (str, optional): _description_. Defaults to "".
        date (str, optional): _description_. Defaults to "".
    """
    
    list_json=os.listdir(PATH)
    list_json_filtered=[json_file for json_file in list_json if (json_file.startswith(num_cam) and (date in json_file)) ]
    
    data_json_dict={}
    frame_json_dict={}
    for json_file in list_json_filtered:
        split = re.split(r'[_]', json_file)
        date_specific = re.split(r'[-]',re.split(r'[.]',split[-1])[0])[-1]        
        
        if "framepassage" in json_file:
            with open(PATH+f"/{json_file}","r") as f:
                frame_json_dict[date_specific]=json.load(f)
        else : 
            with open(PATH+f"/{json_file}","r") as f:
                data_json_dict[date_specific]=json.load(f)

    return data_json_dict,frame_json_dict


def preprocess (data_json) :
    """Calcul et écrit dans les jsons les coordonnées des vecteurs vitesse, seulement pour dictionnaires qui ont la structure suivante :
    num_object : {
        frame : {
        x_center,
        y_center,
        width,
        height,
        class_name
        }
    }

    Args:
        data_json (_type_): _description_

    Returns:
        _type_: _description_
        
    """
    for object in data_json :
        frames = list(data_json[object].keys())
        for i,frame_num in enumerate( data_json[object] ):
            if i != 0 :
                frame = data_json[object][frame_num]
                ante_num = frames[frames.index(frame_num)-1]
                ante = data_json[object][ante_num]
                # Calcul de la vitesse respectivement selon x, y puis la norme 
                frame.append(round((frame[0]-ante[0])/(int(frame_num)-int(ante_num)),3))
                frame.append(round((frame[1]-ante[1])/(int(frame_num)-int(ante_num)),3))
                frame.append(round(np.sqrt(frame[5]**2+frame[6]**2),3))
                
                if i == 1 :
                    frame.append(0)
                else :
                    # calcul de la norme de l'accélération avec la norme de la vitesse ??? 
                    
                    # A priori l'accélération est un vecteur, donc d'abord calcul de ces coordonnées puis de sa norme
                    #frame.append(round((frame[7]-ante[7])/(int(frame_num)-int(ante_num)),3))
                    x_acc = round((frame[5]-ante[5])/(int(frame_num)-int(ante_num)),3)
                    y_acc = round((frame[6]-ante[6])/(int(frame_num)-int(ante_num)),3)
                    frame.append(round(np.sqrt(x_acc**2+y_acc**2),3))

        # recopie la vitesse et accélération
        if len(frames) > 1 :
            data_json[object][frames[0]].append(data_json[object][frames[1]][5])
            data_json[object][frames[0]].append(data_json[object][frames[1]][6])
            data_json[object][frames[0]].append(data_json[object][frames[1]][7])
            data_json[object][frames[0]].append(data_json[object][frames[1]][8])
        else :
            data_json[object][frames[0]].append(0)
            data_json[object][frames[0]].append(0)
            data_json[object][frames[0]].append(0)
            data_json[object][frames[0]].append(0)
    return data_json