import pandas as pd
from datetime import datetime, timedelta

def video_to_date(file_path) :
  file_name = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] # '/content/0-02-202403181514.mkv' -> '0-02-202403181514.mkv'
  date_video = re.split(r'[-.]', video_file)[2] # '0-02-202403181514.mkv' ->  ['0', '02', '202403181514', 'mkv']  ->  '202403181514'
  date_time = datetime.strptime(date_video, "%Y%m%d%H%M").replace(tzinfo=timezone('UTC')) # '202403181514' -> datetime.datetime(2024, 3, 18, 15, 14, tzinfo=<UTC>)
  return date_time

def associate_measure(file_path, time_sample_minutes, df) :
  date_video = video_to_date(file_path)
  time_sample_timedelta = timedelta(minutes=time_sample_minutes)
  # Convert the 'Date' column to datetime if it's not already in datetime format
  df['Date'] = pd.to_datetime(df['Date'])
  # Find the closest measurements
  measure = df[(df['Date'] >= date_video) & (df['Date'] <= date_video + time_sample_timedelta)]
  return measure
