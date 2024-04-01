import numpy as np
from scipy.interpolate import interp1d


def linear_speed_estimate (time, speed) :
  # Convert time from str to int
  time = np.array([int(t) for t in time])

  # Perform linear interpolation for missing values in speed
  f_speed = interp1d(time, speed, kind='linear')

  # Define the range of time for interpolation
  min_time = time.min()
  max_time = time.max()

  # Generate new time values in the range [min_time, max_time]
  new_time = np.arange(min_time, max_time + 1)

  # Interpolate missing values in speed using the new time values
  new_speed = f_speed(new_time)

  return new_time, new_speed

def constant_acc_estimate (time, acc) :
  time = np.array([int(t) for t in time])
  f_acc = interp1d(time, acc, kind='previous')
  min_time = time.min()
  max_time = time.max()
  new_time = np.arange(min_time, max_time + 1)
  new_acc = f_acc(new_time)
  return new_time, new_acc

# Function to get intervals of size 30 containing index n
def get_intervals(array, n, interval_size):
    n = int(n)
    intervals = []

    # Iterate through the array to find intervals centered around index n
    for i in range(interval_size):
        start_index = n - interval_size + 1 +i
        end_index =  n + 1 + i
        if start_index >= 0 and end_index < len(array) :
          intervals.append (array[start_index:end_index] )
    return np.array(intervals)

def nb_actual_measures (frames,interval) :
  return len(np.intersect1d(frames.astype(int),interval))

def max_actual_measures (frames,interval_size,time,frame) :
  intervals = get_intervals(time,frame,interval_size)
  if intervals.size == 0 :
    return 0
  return max ([nb_actual_measures(frames,interval) for interval in intervals])