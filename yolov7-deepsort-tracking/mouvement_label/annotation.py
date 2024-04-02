from mouvement_label.preprocess import preprocess
from mouvement_label.detection_mouvement import frame_by_frame
from mouvement_label.association_video_measurements import associate_measure
from tqdm import tqdm
import numpy as np 

def detection_mouvement(data_json_dict,frame_json_dict):
    experience_object_detected={}
    for date_mouvment_detecte in tqdm(data_json_dict.keys()):
        data_json_dict[date_mouvment_detecte]=preprocess(data_json_dict[date_mouvment_detecte])
        fbf,nb = frame_by_frame(frame_json_dict[date_mouvment_detecte],data_json_dict[date_mouvment_detecte])
        experience_object_detected[date_mouvment_detecte]={"fbf":fbf,"nb":nb}
    return experience_object_detected


def get_signal_motionless_object(data_exp):
    nb_moveless={}
    for date_video in data_exp.keys():
        nb_moveless[date_video]=[]
        for frame in data_exp[date_video]["fbf"].keys():
            nb_moveless[date_video].append(len(data_exp[date_video]["fbf"][frame][1]))
    
    return nb_moveless

def median_filter_time_series(data, window_size=30):
    filtered_data = np.zeros_like(data)
    for i in range(len(data)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(data), i + window_size // 2 + 1)
        neighborhood = data[start_index:end_index]
        filtered_data[i] = np.median(neighborhood)
    return filtered_data

# Fonction pour détecter les segments de valeurs constantes
def segments_valeurs_constantes(serie):
    serie_t=np.array(serie.copy())[1:]
    serie_t_1 = np.array(serie.copy())[:-1]
    
    return serie_t-serie_t_1

def annotate(data_exp):
    annotation={}
    for date in data_exp.keys():
        motion=np.array(list(data_exp[date]["nb"].values()))
        smoothed_time_series=median_filter_time_series(motion, 15)
        segments = segments_valeurs_constantes(smoothed_time_series)
        nb_passage = 0
        for num in segments:
            if num > 0:
                nb_passage += num 

        if np.any(motion):
            annotation[date]=f"{nb_passage} passage(s)"
        else : 
            annotation[date]=f"aucun passage"
    
    return annotation

def add_labels_to_df(df_capteurs,annotation):
    
    # prend les events uniques détectés
    events= set(list(annotation.values()))
    df = df_capteurs.copy(deep=True)
    # initialise l'ensemble des events à 0 pour toutes mesures
    for event in events:
        df[event]=np.zeros(df.Date.shape).T
    # met à 1 les lignes dans les colonnes où les events se sont passés
    for date in annotation.keys():
        print(date)
        mesure_indexes = associate_measure(date,5,df)
        print(mesure_indexes)
        for mesure_index in mesure_indexes:
          df[annotation[date]].iloc[mesure_index]=1
    for index in df.index:
        df["aucun passage"].iloc[index] = int(not(df["1 passage(s)"].iloc[index].item() or df["2 passage(s)"].iloc[index].item() or df["3 passage(s)"].iloc[index].item()))
    
    df_labels_added=df.dropna(axis=0).reset_index(drop=True, inplace=False)
    
    return df_labels_added