import numpy as np
from mouvement_label.estimation import linear_speed_estimate,constant_acc_estimate,get_intervals,max_actual_measures


def detect_moving_frame (frames_json, data_json, frame, speed_threshold = 3, mean_speed_threshold = 5, frame_ratio = 0.3, time_interval = 15, acc_threshold = 2) :
  moving_objects = []
  motionless_objects = []
  for object in frames_json[frame] :
    object_stat = data_json[str(object)]
    object_frame = object_stat[frame]
    time = np.array(list(object_stat.keys()))
    speed = np.array([object_stat[f][7] for f in time])
    acceleration = np.array([object_stat[f][8] for f in time])
    t,v = linear_speed_estimate (time,speed)
    _,a = constant_acc_estimate (time,acceleration)
    speed_intervals = get_intervals(v, int(frame), time_interval)
    a = speed_intervals
    acc_intervals = get_intervals(a, int(frame), time_interval)
    if  (max_actual_measures (time,time_interval,t,frame) > frame_ratio*time_interval) and (len(v) > time_interval) and np.any(np.all(speed_intervals != 0, axis = -1))  and np.any(np.median(speed_intervals,axis = 1) > mean_speed_threshold) and np.all(acc_intervals < acc_threshold) :
      moving_objects.append(object)
    else :
      motionless_objects.append(object)
  return moving_objects, motionless_objects

def frame_by_frame (frames_json, data_json) :
  moving_by_frames = {}
  nb_moving_by_frame = {}
  for frame in frames_json :
    moving_by_frames[frame] = detect_moving_frame (frames_json, data_json, frame)
    nb_moving_by_frame[frame] = len(moving_by_frames[frame][0])
  return moving_by_frames, nb_moving_by_frame
