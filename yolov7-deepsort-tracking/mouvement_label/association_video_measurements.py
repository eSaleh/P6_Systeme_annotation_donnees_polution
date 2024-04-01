import pandas as pd
from datetime import datetime, timedelta,timezone
import numpy as np


def video_to_date(date_video) :
  #file_name = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] # '/content/0-02-202403181514.mkv' -> '0-02-202403181514.mkv'
  #date_video = re.split(r'[-.]', file_path)[2] # '0-02-202403181514.mkv' ->  ['0', '02', '202403181514', 'mkv']  ->  '202403181514'
  date_time = datetime.strptime(date_video, "%Y%m%d%H%M%S").replace(tzinfo=timezone(timedelta(hours=1))) # '202403181514' -> datetime.datetime(2024, 3, 18, 15, 14, tzinfo=<CTE>)
  return date_time

def associate_measure(date_video, time_sample_minutes, df) :
  date_video = video_to_date(date_video)
  time_sample_timedelta = timedelta(minutes=time_sample_minutes)
  # Convert the 'Date' column to datetime if it's not already in datetime format
  df['Date'] = pd.to_datetime(df['Date'])
  # Find the closest measurements and give indexes
  measure = df[(df['Date'] >= date_video) & (df['Date'] <= date_video + time_sample_timedelta)].index
  return measure

def association(annotation_json,df,temps_suspension_pm2_5=10):
  #prend les events uniques détectés
  events= set(list(annotation_json.values()))
  #initialise l'ensemble des events à 0 pour toutes mesures
  for event in events:
    df[event]=np.zeros(df.Date.shape).T

  #trouve les mesures les plus proches et met à 1 les colonnes d'event correspondant aux mesures
  for date in annotation_json.keys():
    mesure_indexes = associate_measure(date,temps_suspension_pm2_5,df)
  
    for mesure_index in mesure_indexes:
      df.loc[annotation_json[date],mesure_index]=1

  return df