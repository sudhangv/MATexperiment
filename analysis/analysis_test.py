import os
import sys
from functools import partial
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
from global_folder.myplotsty import *
from global_folder.my_helpers import *

PUMP_FREQUENCY = 384228.6
REPUMP_FREQUENCY = 384228.6 + 6.56
SAMPLE_RATE = 2000
FREQVSVOLT = 221.0 
FREQVSCURR = 1.13

# TODO: find a better place for this
EXP_FOLDER =r'C:\Users\svars\OneDrive\Desktop\UBC Lab\CATExperiment\CATMeasurements'
MEASURE_FOLDER = os.path.join(EXP_FOLDER, 'testPArun16')
WDATA_FOLDER =os.path.join(MEASURE_FOLDER, 'testPArun16.csv')

# TODO: maybe make a run analysis class out of this?
def dump():
	fig, ax = plt.subplots()
	plot_results(ax, df, max_freq= 384219., fmt='o', mfc='red', save_folder=MEASURE_FOLDER)
	plt.savefig(os.path.join(MEASURE_FOLDER, 'ratio_vs_freq.png'))

	
	collect_plots(MEASURE_FOLDER, os.path.join(MEASURE_FOLDER, 'collected_plots'), 'deloadPhase.png')
 
	#*-----------------------
	#* SINGLE RUN
 	#*----------------------- 
  
	MEASURE_FOLDER = os.path.join(EXP_FOLDER, 'testPArun9')
	df = get_data_frame(MEASURE_FOLDER)
	df.dropna(inplace=True)
	#freqs = plot_results(df, 384201., save_folder=MEASURE_FOLDER)
 
	data = df.dropna()
	freqs = ((384201-PUMP_FREQUENCY)-(data['tempV']-df['tempV'].min())*FREQVSVOLT- (data['currV']-df['currV'].min())*FREQVSCURR)
	fig, ax = plt.subplots()
	plot_spline_fit(ax=ax, x=freqs, y=data['ratio'], yerr=data['ratioErr'], 
				 s=0.1, save_folder=MEASURE_FOLDER, 
				 mfc='red', color='black', 
				 title='Trap Depth = 1.99 K')
 
	# *-----------------------
	# * MULTIPLE RUN COMPARISON
 	# *-----------------------
	folders = [os.path.join(EXP_FOLDER, path ) for path in ['testPARun11', 'testPARun12', 'testPARun13', 'testPARun14']]
	dfs = [get_data_frame(measure_folder, cache_all=True) for measure_folder in folders]

	colors = ['red', 'dodgerblue', 'green', 'grey']
	max_freqs = [384178.881, 384178.593,384179.068, 384184.731]
	#max_freqs=[384178.599, 384179.091, 384184.733]

	zipped_data = list(zip(dfs, max_freqs))
	fig, ax = plt.subplots()
	for i, (df, max_freq)  in enumerate(zipped_data[:3]):
		data = df.dropna()
		freqs = ((max_freq-PUMP_FREQUENCY)-(data['tempV']-data['tempV'].min())*FREQVSVOLT- (data['currV']-data['currV'].min())*FREQVSCURR)
		color = colors[i]
		ax=plot_spline_fit(ax, x=freqs, y=data['ratio'], scolor=color, mfc=color,color=color, s=0.02, ms=5)

	#*-----------------------
	#* PARSING WAVEMETER DATA
 	#*-----------------------

	MEASURE_FOLDER = os.path.join(EXP_FOLDER, 'testPArun14')
	WDATA_FOLDER =os.path.join(MEASURE_FOLDER, 'testPArun14.csv')
	freq_data, max_freq, min_freq = add_wavemeter_data('', WDATA_FOLDER)
	data = freq_data[:]
	levels = staircase_fit(data)

	data = get_data_frame(MEASURE_FOLDER)
	data.dropna(inplace=True)
	freqs = ((max_freq)-(data['tempV']-data['tempV'].min())*FREQVSVOLT- (data['currV']-data['currV'].min())*FREQVSCURR)
	plt.plot(freqs)
 
	#*-----------------------
	#* GETTING DEPTH_RATIO DATAFRAME
 	#*-----------------------
	MEASURE_FOLDER = r'C:\Users\svars\OneDrive\Desktop\UBC Lab\CATExperiment\CATMeasurements\relScatRate'
	depth_ratios_df = get_rel_scattering_df(MEASURE_FOLDER)
	df_slice = depth_ratios_df[(depth_ratios_df['pa1']==1.85) & (depth_ratios_df['pd1']==84) & (depth_ratios_df['pd2'] == 84)]
	x = df_slice['pa2']
	y = df_slice['depth_ratio']
	plt.plot(x,y, 'o')

	#*-----------------------
	#* MEGA_RUN
 	#*-----------------------	
	MEASURE_FOLDER = os.path.join(EXP_FOLDER, 'testPARunMega')
	df = get_data_frame(MEASURE_FOLDER)
	df.dropna(inplace=True)

	df_grouped = df.groupby(by='pump_reference')
	min_ratios = df_grouped['ratio'].min()

	groups = dict(list(df_grouped))
	dfs = [df for df in groups.values()]
 
	max_freqs = [384182.5]*15
	#max_freqs=[384178.599, 384179.091, 384184.733]
	zipped_data = list(zip(dfs, max_freqs))
	fig, ax = plt.subplots()
	for i, (df, max_freq)  in enumerate(zipped_data[:]):
		data = df.dropna()
		freqs = ((max_freq-PUMP_FREQUENCY)-(data['tempV']-data['tempV'].min())*FREQVSVOLT- (data['currV']-data['currV'].min())*FREQVSCURR)
		ax=plot_spline_fit(ax, x=freqs, y=data['ratio'], yerr=data['ratioErr'],scolor=f'C{i}', mfc=f'C{i}',color=f'C{i}', s=0.0, ms=5, figsize=(10, 10), label=f"Pump Amplituide = { df.iloc[10]['pump_reference'] :.2f}", linewidth=2.5)

	plt.legend()

	plt.savefig(os.path.join(MEASURE_FOLDER, 'lossFeatures.png'))
 
	x = [(180-2*df['pump_AOM_freq'].mean()) for df in dfs]
	y = [(df['ratio'].max() - df['ratio'].min()) for df in dfs]
	plt.plot( x, y ,'-o', label=fr"$\delta$ = {180 - 2*dfs[0]['pump_AOM_freq'].mean()} MHz")
	plt.xlabel(r'$\delta$ (MHz)')
	plt.ylabel(r'SNR $ = V_{ss, off} - V_{ss, on}$ ')
 
	x = [df['pump_reference'].mean() for df in dfs]
	y = [(df['ratio'].max() - df['ratio'].min()) for df in dfs]
	plt.plot( x, y ,'-o', label=f"Pump Amplitude = {df['pump_reference'].mean()}")
	plt.xlabel('Pump Amplitude')
	plt.ylabel(r'SNR $ = V_{ss, off} - V_{ss, on}$ ')
	plt.title(fr"$\delta $ ={180-2*df['pump_AOM_freq'].mean()}")
	#*-----------------------
	#* MULTIPLE MEGARUN
	#*-----------------------
 
	folders = [os.path.join(EXP_FOLDER, path ) for path in ['testPArunMega', 'testPArunMega2']]
	dfs_mega = [get_data_frame(measure_folder, cache_all=True).dropna() for measure_folder in folders]

	dfs_grouped = [df_mega.groupby(by='pump_reference') for df_mega in dfs_mega]
	min_ratios = [df_grouped['ratio'].min() for df_grouped in dfs_grouped]

	groupss = [dict(list(df_grouped)) for df_grouped in dfs_grouped]
	dfs = [ [df for df in groups.values()] for groups in groupss]
 
	for row in dfs:
		x = [df['pump_reference'].mean() for df in row]
		y = [(df['ratio'].max() - df['ratio'].min())/df['motSS'].std() for df in row]
		plt.plot( x, y ,'-o', label=fr"$\delta$ = {180 - 2*row[0]['pump_AOM_freq'].mean()} MHz")
		plt.xlabel('Pump Amplitude')
		plt.ylabel(r'SNR $ = \frac{V_{ss, off} - V_{ss, on}}{\sigma_{V,off}}$ ')
	plt.legend()

def freq_misc():
	WDATA_FOLDER = r'C:\Users\svars\OneDrive\Desktop\UBC Lab\CATExperiment\CATMeasurements\CATcurrTestrun3.csv'
	freq_data = add_wavemeter_data('', WDATA_FOLDER)
	levels = staircase_fit(freq_data[0], peak_height=0.2, distance=50, data_offset=1, window_size=1) 

	plt.close()
 
	x = np.linspace(0, 4.9, 25)
	y = levels
 
	plt.plot(levels, '-o')
	plt.title('Levels plot')
	
	m, b, fit_line = my_linear_fit(x, y)
def save_fit_results(run_path, plot=False):
	filename = os.path.join(run_path, 'data.csv')
	bkfilename = os.path.join(run_path, 'data_oldPD.csv')
	settingsname = os.path.join(run_path, 'Settings.txt')
	
	dataHost = MTdataHost(SAMPLE_RATE)
	dataHost.loadCATdata(fileName=filename, settingsName=settingsname)
	dataHost.CATbackgroundData(bkfilename)
	
	dataHost.setAllCAT(0.002)
	resultDict = dataHost.getResults(run_path, store=True)
 
	if plot:
		dataHost.storeFits(run_path, combined=True, separate=True)
  
	return resultDict, dataHost.settings
		
def get_timestamp(run_path):
	timestamp = datetime.strptime(os.path.split(run_path)[-1].split('_')[0], dateformat)
	return timestamp

def extract_fit(run_path, plot=True, cache_failed=True, cache_all=True):
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
			fit_results, settings = save_fit_results(run_path, plot=plot)
		except Exception as e:
			print(traceback.format_exc())
			print("Fitting ERROR at ", os.path.basename(run_path), '\n')

			with open(MAT_fit_cache_path, 'w') as f:
				f.write(str('MAT fit failed'))
		
	else:
		print("Accessing cached results from :", os.path.basename(run_path))
  
		fit_results = open(MAT_fit_cache_path, 'r').read()

		if fit_results == 'MAT fit failed':
			if not cache_failed:
				# fit regardless of cached result
				try: 
					fit_results, settings = save_fit_results(run_path, plot=plot)
				except Exception as e:
					print(traceback.format_exc())
					print("Fitting ERROR at ", os.path.basename(run_path), '\n')
	
					with open(MAT_fit_cache_path, 'w') as f:
						f.write(str('MAT fit failed'))	
			else:
				print("Failed fit at :", os.path.basename(run_path))
				fit_results = {}

		else:
			fit_results = eval(open(MAT_fit_cache_path, 'r').read())
   
			settingsname = os.path.join(run_path, 'Settings.txt')
			settings = eval(open(settingsname, 'r').read())
   
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
			rows = list(tqdm(p.imap(partial(get_row, **kwargs), run_path_arr), total=len(run_path_arr)))
	
	else:
		for run_path in tqdm(run_path_arr):
		   rows.append(get_row(run_path, **kwargs))

	return pd.DataFrame.from_dict(rows)

def add_wavemeter_data(df, wmeter_csv_path, window_size=100, num_rows=50):


	"""Extract unique frequnecy values from wavemeter data
	Returns:
		unique_levels (list): unique frequency values in wavemeter data
	"""
 
	# TODO: modify dataframe in place with frequency data
 
	wdata = pd.read_csv(wmeter_csv_path, skiprows=2)
	wdata.dropna(inplace=True)
	freq_data = np.array(wdata.iloc[:, 0])
	try:
		freq_data = np.array([float(item) for item in freq_data if item.replace('.','').isdigit()])
	except Exception as e:
		print(e)
 
	max_freq = freq_data.max()
	min_freq = freq_data.min()

	return freq_data, max_freq, min_freq

def plot_results(ax, dfs, max_freq, min_freq=0.0, mfc='red', fmt='o', ms=5, save_folder=False, xscale=1.0, yscale=1.0, **kwargs):
	FREQVSVOLT = 221.0 
	FREQVSCURR = 1.13
	
	if not type(dfs) == list:
		freqs = ((max_freq-PUMP_FREQUENCY)-(dfs.dropna()['tempV']-dfs.dropna()['tempV'].min())*FREQVSVOLT- (dfs.dropna()['currV']-dfs.dropna()['currV'].min())*FREQVSCURR)*xscale
		dfs=[dfs]
  
	plt.gcf().set_dpi(300)
	
	for df in dfs:
		df = df.dropna()
		ax.errorbar(freqs, 
			   df['ratio']*yscale,
			   yerr=df['ratioErr'],
			   fmt=fmt, mfc=mfc, color='black', ms=ms, **kwargs)
		ax.set_ylabel(r'$\mathbf{\frac{V_{ss, cat}}{V_{ss}}} $ ', **labeldict)
		ax.set_xlabel(r'$\Delta $ (GHz)', **labeldict)
	
	if save_folder:
		plt.savefig(os.path.join(save_folder, 'ratio_vs_freq.png'))
	
	return freqs, ax
  
	#return plt.gca(), plt.gcf()
	#plt.show()

def plot_spline_fit(ax, x, y, s=1, yerr=None, color='black', scolor='black',figsize=(12,5), save_folder=None, title='',alpha=0.5,dpi=200, label='plot',**kwargs):
	from scipy.interpolate import splev, splrep
	xnew = np.linspace(min(x), max(x), 3*len(x) )

	y = [b for a,b in sorted(zip(x,y), key=lambda pair: pair[0])]
	if yerr is not None:
		yerr = [b for a,b in sorted(zip(x,yerr), key=lambda pair: pair[0])]
 
	x = sorted(x)

	spl = splrep(x, y, s=s)
	ynew = splev(xnew, spl)

	
	plt.gcf().set_dpi(dpi)
	plt.gcf().set_size_inches(figsize)
	
	if yerr is not None:
		ax.errorbar(x, y, yerr=yerr, fmt='o', **kwargs)
	else:
		ax.plot(x,y, 'o', **kwargs)
	ax.plot(xnew, ynew, '-', color=scolor, alpha=alpha, label=label, **kwargs)
 
	ax.set_ylabel(r'$\mathbf{\frac{V_{ss, cat}}{V_{ss}}} $ ', **labeldict)
	ax.set_xlabel(r'$\Delta $ (GHz)', **labeldict)
 
	ax.set_title(title, **titledict)
	if save_folder:
		plt.savefig(os.path.join(save_folder, 'spline_ratio_vs_freq.png'))
	
	return ax
 
def plot_polyfit(x_data, y_data, spline_degree):
	
	coefficients = np.polyfit(x_data, y_data, spline_degree)

	x_interp = np.linspace(min(x_data), max(x_data), 100)
	y_interp = np.polyval(coefficients, x_interp)

	plt.scatter(x_data, y_data, label='Original Data')
	plt.plot(x_interp, y_interp, label='Polynomial Interpolation (Degree={})'.format(spline_degree))

def collect_plots(source, destination, plot_name):
	print(f'Collecting plots from {os.path.basename(source)}')
	import shutil

	os.makedirs(destination, exist_ok=True)

	plot_files = []
	for root, dirs, files in os.walk(source):
		
		for file in files:
			if file == plot_name:
				plot_files.append(os.path.join(root, file))

	for i, plot_file in enumerate(plot_files, start=0):
		new_filename = f'{i}{plot_name}'
		destination_path = os.path.join(destination, new_filename)
		shutil.copy(plot_file, destination_path)

def create_GIF(images_folder, image_name):
	import imageio
	with imageio.get_writer(os.path.join(images_folder, f'{image_name}movie.gif'), mode='I', duration=0.5) as writer:
		for filename in os.listdir(images_folder):
			if image_name in filename:
				image = imageio.imread(os.path.join(images_folder, filename))
				writer.append_data(image)

def staircase_fit(data, peak_height=0.1, distance=100, data_offset=1, window_size=1, inc_final_peak=True):
	def moving_average(arr, window_size):
		weights = np.ones(window_size) / window_size
		return np.convolve(arr, weights, mode='valid')

	convdata1 = moving_average(data, window_size)
	convdata2 = moving_average(data[data_offset+1:], window_size )
	final = convdata1[:len(convdata2)]-convdata2
	# plt.plot(data)
	# plt.plot(convdata1)
	# plt.plot(convdata2)
	# plt.show()
	# plt.plot(final)

	
	from scipy.signal import find_peaks
	x= final
	peaks, _ = find_peaks(x, height=peak_height, distance=distance)
	peaks = np.insert(peaks, 0, 0)
	levels = []
	plot_arr = []
	for i, peak in enumerate(peaks):
		
		if i < len(peaks) - 1:
			temp = data[ peaks[i]:peaks[i+1] ]
			plot_arr.extend( np.ones_like(temp)*np.mean(temp))
			levels.append(np.mean(temp))
	if inc_final_peak:
		temp = data[peaks[-1]:]
		levels.append(np.mean(temp))
		plot_arr.extend( np.ones_like(temp)*np.mean(temp))
  
	plt.plot(x)
	plt.plot(peaks, x[peaks], "x")
	plt.plot(np.zeros_like(x), "--", color="gray")
	plt.show()
	
	plt.plot(data)
	plt.plot(np.ravel((plot_arr)))
	plt.show()
	plt.close()
	plt.plot(np.array(levels)[np.where(abs(np.diff(levels))>0.05)[0]], 'o', ms=5)
 
	return levels

def load_single_run(run_path):
	filename = os.path.join(run_path, 'data.csv')
	bkfilename = os.path.join(run_path, 'data_oldPD.csv')
	settingsname = os.path.join(run_path, 'Settings.txt')
	
	dh1 = MTdataHost(SAMPLE_RATE)
	dh1.loadCATdata(fileName=filename, settingsName=settingsname)
	
	return dh1
	



if __name__ == '__main__':
	# run_path = r"C:\Users\svars\OneDrive\Desktop\UBC Lab\CATExperiment\CATMeasurements\testPArun9\16-53-10"
	# filename = os.path.join(run_path, 'data.csv')
	# bkfilename = os.path.join(run_path, 'data_oldPD.csv')
	# settingsname = os.path.join(run_path, 'Settings.txt')
	
	# dh1 = MTdataHost(SAMPLE_RATE)
	# dh1.loadCATdata(fileName=filename, settingsName=settingsname)
	
	# dh1.setAllCAT(0.002)
 
 	#dh1.CATbackgroundData(bkfilename)
	
	pass
