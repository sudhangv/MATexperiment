from  datetime import datetime, timedelta
dateformat = "%H-%M-%S"
from collections import deque
from functools import partial
from multiprocessing import Pool
from os.path import join 
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..')) #adds directory below as valid path
import traceback

import scipy.constants as spc
from lmfit import Model, create_params

from scipy.integrate import odeint
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['axes.grid'] = True
plt.rcParams['grid.linestyle'] = '--'

from MT_class_PID_new import MTdataHost

# comment these out, and set titleDict= {}
from global_folder.myplotsty import *
from global_folder.my_helpers import *

# SOME CONSTANTS

PUMP_FREQUENCY   = 384228.6
REPUMP_FREQUENCY = 384228.6 + 6.56
SAMPLE_RATE      = 2000
FREQVSVOLT       = 221.0
FREQVSCURR       = 1.13


# TODO: find a better place for this
EXP_FOLDER =r'C:\Users\svars\OneDrive\Desktop\UBC Lab\CATExperiment\CATMeasurements'
MEASURE_FOLDER = os.path.join(EXP_FOLDER, 'testPArun16')
WDATA_FOLDER =os.path.join(MEASURE_FOLDER, 'testPArun16.csv')



def save_fit_results(run_path, plot=False, bkfile=False, 
					 CATbaseline=True, MOTbaseline=True,
					 initRFit=True, loadFit=True,deloadFit=True,reloadFit=True,
					 storeFitResults=True):
	"""_summary_

	Args:
		run_path (str)                  : absolute path of run folder
		plot (bool, optional)           : plot fits to data. Defaults to False.
		bkfile (bool, optional)         : load background data file. Defaults to False.
		CATbaseline (bool, optional)    : fit CAT baseline. Defaults to True.
		MOTbaseline (bool, optional)    : fit MOT baseline. Defaults to True.
		initRFit (bool, optional)       : fit initial Loading rate. Defaults to True.
		loadFit (bool, optional)        : fit MOT loading curve. Defaults to True.
		deloadFit (bool, optional)      : fit MOT deloading curve. Defaults to True.
		reloadFit (bool, optional)      : fit MOT reloading curve. Defaults to True.
		storeFitResults (bool, optional): store fit results in a text file in run_folder. Defaults to True.

	Returns:
		(resultDict, settings): dictionaries of the fit results and run settings
	"""
	filename = os.path.join(run_path, 'data.csv')
	bkfilename = os.path.join(run_path, 'data_oldPD.csv')
	settingsname = os.path.join(run_path, 'Settings.txt')
	
	dataHost = MTdataHost(SAMPLE_RATE)
	dataHost.loadCATdata(fileName=filename, settingsName=settingsname)
	
	if bkfile:
		dataHost.CATbackgroundData(bkfilename)
	
	#dataHost.setAllCAT(0.002)
	if CATbaseline:
		dataHost.setCATbaseline(0.002)
	if MOTbaseline:
		dataHost.setBaseline(0.002)
	if loadFit:
		dataHost.setLoading(0.002)
	if initRFit and loadFit:
		dataHost.initFit, dataHost.initX = dataHost.setInitialLoad(0.002)
	if deloadFit:
		dataHost.setDeloading(0.002)
		dataHost.plotDeloadFit(run_path)# TODO: currently just stores the deloading times and voltages
	if reloadFit:
		dataHost.setReloadVolt(0.002)

	if CATbaseline and MOTbaseline and loadFit and reloadFit:
	# steady state ratio fraction
		dataHost.ratio = dataHost.reloadVolt / dataHost.motSS

		dataHost.ratioErr = dataHost.ratio * ((dataHost.reloadVoltErr/dataHost.reloadVolt)**2 + (dataHost.motSSErr/dataHost.motSS)**2)**(0.5)
	
		# if abs(dataHost.ratioErr / dataHost.ratio) > 0.1:
		#     dataHost.ratioErr = abs(0.015*dataHost.ratio)
		if dataHost.ratioErr < 0.001:
			dataHost.ratioErr = 0.001
		if dataHost.ratio < 0:
			dataHost.ratio = 0
			
		# TODO: this information is useless
	print('File loaded: RFmin = {} MHz, t_mt = {:.3f} s.'.format(dataHost.settings['fmin'], dataHost.settings['wait_mtrap']))
	
	resultDict = dataHost.getResults(run_path, store=storeFitResults)
 
	if plot:
		dataHost.storeFits(run_path, combined=True, separate=True)
  
	return resultDict, dataHost.settings
		
def get_timestamp(run_path):
	"""extract timestamp (H-M-S) from run folder

	Args:
		run_path (str): absolute path of run folder

	Returns:
		timestamp: format: H-M-S
	"""
	timestamp = datetime.strptime(os.path.split(run_path)[-1].split('_')[0], dateformat)
	return timestamp

def extract_fit(run_path, plot=True, cache_failed=True, cache_all=True, **kwargs):
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
			fit_results, settings = save_fit_results(run_path, plot=plot, **kwargs)
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

	"""wrapper for extract_fit which returns a row of the final dataframe. The columns are the run settings and fitted quanitities for each run. 
 
	Returns:
		row (dict): data corresponding to a single run
	"""
	fit_results, settings, timestamp = extract_fit(run_path, **kwargs)
	row = {**fit_results, **settings, **{'timestamp':timestamp}}
	return row

   
def get_data_frame(data_dir, parallel=True, in_process_run=False, **kwargs):
	
	"""Higher level function that fits the data, plots the fits and returns a dataframe with the results

	Arguments:
		data_dir (str): Full path to the measurement directory
		parallel (bool): Analyze in parallel (default = True)
		in_prcoess_run (bool): analyze till penultimate run, useful if analuzing WHILE taking measurements
  
	Returns:
		df (DataFrame): data for all the runs in data_dir
	"""
   
	run_path_arr = []
	rows = []
	
	for relative_path in os.listdir(data_dir):
		run_path_arr.append(os.path.join(data_dir, relative_path))
  
	if in_process_run:
		run_path_arr.pop()

	run_path_arr = sorted(run_path_arr)
	
	if parallel:
		# here you can increase "4" to the number of cores in your system
		with Pool(4) as p:
			rows = list(tqdm(p.imap(partial(get_row, **kwargs), run_path_arr), total=len(run_path_arr)))
	
	else:
		for run_path in tqdm(run_path_arr):
	  		rows.append(get_row(run_path, **kwargs))
	return pd.DataFrame.from_dict(rows)


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

def plot_spline_fit(ax, x, y, s=1, yerr=None, color='black', scolor='black',figsize=(12,5), save_folder=None, title='',alpha=0.5,dpi=200, label='plot', fig=None,**kwargs):
	from scipy.interpolate import splev, splrep
	xnew = np.linspace(min(x), max(x), 3*len(x) )

	y = [b for a,b in sorted(zip(x,y), key=lambda pair: pair[0])]
	if yerr is not None:
		yerr = [b for a,b in sorted(zip(x,yerr), key=lambda pair: pair[0])]
 
	x = sorted(x)


	spl = splrep(x, y, s=s)
	ynew = splev(xnew, spl)

	if fig is None:
		plt.gcf().set_dpi(dpi)
		plt.gcf().set_size_inches(figsize)
	else:
		fig.set_dpi(dpi)
		fig.set_size_inches(figsize)
	if yerr is not None:
		ax.errorbar(x, y, yerr=yerr, fmt='o', color=color, **kwargs)
	else:
		ax.plot(x,y, 'o', **kwargs)
	ax.plot(xnew, ynew, '-', color=scolor, alpha=alpha, label=label, **kwargs)
 
	ax.set_ylabel(r'$\mathbf{\frac{V_{ss, cat}}{V_{ss}}} $ ')
	ax.set_xlabel(r'$\Delta $ (GHz)')
 
	ax.set_title(title, **titledict)
	if save_folder:
		plt.savefig(os.path.join(save_folder, 'spline_ratio_vs_freq.png'))
	
	return ax
 

def load_single_run(run_path):
	filename = os.path.join(run_path, 'data.csv')
	bkfilename = os.path.join(run_path, 'data_oldPD.csv')
	settingsname = os.path.join(run_path, 'Settings.txt')
	
	dh1 = MTdataHost(SAMPLE_RATE)
	dh1.loadCATdata(fileName=filename, settingsName=settingsname)
	
	return dh1
	
"""
======================================================
		MOSTLY UNUSED CODE FROM HEREON
======================================================
"""
def load_mega_run(MEASURE_FOLDER, groupbyKey, titleKey, plot=True, save_plots=False, max_freq=384182.5, **kwargs):
	# TODO: use the plot flag to do something?
	
	df = get_data_frame(MEASURE_FOLDER,
						**kwargs)
	dfc= df.copy()
	#df.dropna(inplace=True)

	df_grouped = df.groupby(by=groupbyKey)
	min_ratios = df_grouped['ratio'].min()

	groups = dict(list(df_grouped))
	dfs = [df for df in groups.values()]

	# plotting ratio vs freq
	max_freqs = [max_freq]*len(dfs)
	zipped_data = list(zip(dfs, max_freqs))
	fig1, ax = plt.subplots()
	for i, (df, max_freq)  in enumerate(zipped_data[:]):
		data = df
		freqs = ((max_freq-PUMP_FREQUENCY)-(data['tempV']-data['tempV'].min())*FREQVSVOLT- (data['currV']-data['currV'].min())*FREQVSCURR)
		ax=plot_spline_fit(ax, x=freqs, y=data['ratio'], yerr=data['ratioErr'],scolor=f'C{i}', mfc=f'C{i}',color=f'C{i}', s=0.0, ms=5, figsize=(10, 10), label=f"{groupbyKey} = { data.iloc[10][groupbyKey] :.2f}", linewidth=2.5)


	plt.legend()
	
	plt.title(f'Loss Features, {titleKey} = {data[titleKey].mean():.2f}', **titledict)
	plt.show()

	#---------------------------------------------------
	fig2 = plt.figure(2)
	x = [df[groupbyKey].mean() for df in dfs]
	y = [(df['ratio'].max() - df['ratio'].min()) for df in dfs]
	plt.plot( x, y ,'-o')
	plt.xlabel(groupbyKey)
	plt.ylabel(r'SNR $ = V_{ss, off} - V_{ss, on}$ ')
	plt.title(f'SNR Plot, {titleKey} = {df[titleKey].mean():.2f}', **titledict)
	plt.show()
	
	#---------------------------------------------------------
	fig3=plt.figure(3)
	for i, df  in enumerate(dfs[:]):
		data = df
		freqs = ((384182.5-PUMP_FREQUENCY)-(data['tempV']-data['tempV'].min())*FREQVSVOLT- (data['currV']-data['currV'].min())*FREQVSCURR)
		betaPAs = [a for a,b in sorted(zip(data['betaPA'], freqs), key=lambda pair:pair[1])]
		freqs = sorted(freqs)
		plt.plot(freqs, betaPAs, 'o-', ms=5, label=f"{groupbyKey}={data.iloc[10][groupbyKey]:.2f}")
	plt.legend()
	plt.xlabel(r'$\Delta$ (GHz)')
	plt.ylabel(r'$\beta_{\mathrm{eff}}$ ')
	plt.title(f'2-body Decay Plot, {titleKey} = {df[titleKey].mean():.2f}', **titledict) 
	plt.show()

	if save_plots:
		fig1.savefig(os.path.join(MEASURE_FOLDER, 'lossFeatures.png'))
		fig2.savefig(join(MEASURE_FOLDER, 'SNRplot.png'), dpi=200)
		fig3.savefig(join(MEASURE_FOLDER, 'betaVsFreq.png'), dpi=200) 
	return dfc


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
