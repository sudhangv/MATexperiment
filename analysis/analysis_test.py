import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..')) #adds directory below as valid path
from  datetime import datetime, timedelta
dateformat = "%H-%M-%S"
from collections import deque
import traceback
from multiprocessing import Pool
from tqdm import tqdm

import scipy.constants as spc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'

from MT_class_PID_new import MTdataHost

SAMPLE_RATE = 2000
# TODO: find a better place for this
MEASURE_FOLDER =r'C:\Users\svars\OneDrive\Desktop\UBC Lab\CATExperiment\CATMeasurements\testCATRun4'

def save_fit_results(run_path, plot=False):
	filename = os.path.join(run_path, 'data.csv')
	settingsname = os.path.join(run_path, 'Settings.txt')
	
	dataHost = MTdataHost(SAMPLE_RATE)
	dataHost.loadCATdata(fileName=filename, settingsName=settingsname)
	
	dataHost.setAllCAT(0.002)
	resultDict = dataHost.getResults(run_path, store=True)
 
	if plot:
		dataHost.storeFits(run_path, combined=True, separate=True)
  
	return resultDict, dataHost.settings
		
def get_timestamp(run_path):
	timestamp = datetime.strptime(os.path.split(run_path)[-1].split('_')[0], dateformat)
	return timestamp

def extract_fit(run_path, plot=True, cache_failed=True, cache_all=False, time_from_prom=True):
	"""Gather relevant data from each measurement run

	Args:
		run_path : absolute path to the run directory
		plot (bool, optional): plot fits. Defaults to True.
		cache_failed (bool, optional): Cache failed fits. If false, refit. Doesn't refit non-failed fits. Defaults to True.
		cache_all (bool, optional): If false, ignore any cached fit_results. Defaults to True.

	Returns:
		a 3-tuple (fit_results, settings, timestamp)
	"""
	fit_results, settings, timestamp = {}, {}, None
	
	if not os.path.isdir(run_path):
		return fit_results, settings, timestamp  # directory is not a run directory

	try:
		timestamp = get_timestamp(run_path)
  
  	# TODO: specify which error to catch
	except Exception as e:
		print("Error extracting timestamp from: ", run_path)
		print(traceback.format_exc())
	
	MAT_fit_cache_path = os.path.join(run_path, 'resultDict.txt')
	if not os.path.exists(MAT_fit_cache_path) or not cache_all:
		
		try: 
			fit_results, settings = save_fit_results(run_path, plot=True)
		except Exception as e:
			print(traceback.format_exc())
			print("Fitting ERROR at ", os.path.basename(run_path), '\n')

			with open(os.path.join(run_path, 'fitFailed.txt'), 'w') as f:
				f.write(str('MAT fit failed'))
		
	else:
		pass
 	# TODO: caching results
	return fit_results, settings, timestamp

def get_row(run_path, **kwargs):
			
   fit_results, settings, timestamp = extract_fit(run_path, **kwargs)
   row = {**fit_results, **settings, **{'timestamp':timestamp}}
   
   return row

   
def get_data_frame(data_dir, parallel=True, in_process_run=False, **kwargs):
   
	run_path_arr = []
	rows = []
	
	for relative_path in os.listdir(data_dir):
		run_path_arr.append(os.path.join(data_dir, relative_path))
	
	if in_process_run:
		run_path_arr.pop()

	run_path_arr = sorted(run_path_arr)
	
	if parallel:
		with Pool(4) as p:
			rows = list(tqdm(p.imap(get_row, run_path_arr), total=len(run_path_arr)))
	
	else:
		for run_path in tqdm(run_path_arr):
		   rows.append(get_row(run_path, **kwargs))

	return pd.DataFrame.from_dict(rows)

def add_wavemeter_data(df, wmeter_csv_path):

	"""Extract unique frequnecy values from wavemeter data

	Returns:
		unique_levels (list): unique frequency values in wavemeter data
	"""
 
	# TODO: modify dataframe in place with frequency data
 
	wdata = pd.read_csv(wmeter_csv_path, skiprows=2)
	window_size = 2
	rolling_mean = wdata.iloc[:, 0].rolling(window=window_size, min_periods=1).mean()
 
	threshold = 0.5 
	change_points = wdata.iloc[:, 0][np.abs(wdata.iloc[:, 0] - rolling_mean) > threshold]
	unique_levels = change_points.unique()

	#  DEBUGGING 
	return unique_levels, wdata

	
if __name__ == '__main__':
	run_path = r"C:\Users\svars\OneDrive\Desktop\UBC Lab\CATExperiment\CATMeasurements\testCATRun3\16-07-11"
	filename = os.path.join(run_path, 'data.csv')
	settingsname = os.path.join(run_path, 'Settings.txt')
	
	dataHost = MTdataHost(SAMPLE_RATE)
	dataHost.loadCATdata(fileName=filename, settingsName=settingsname)
	
	#dataHost.setAllCAT(0.002)
	#resultDict = dataHost.getResults(run_path, store=True)
 