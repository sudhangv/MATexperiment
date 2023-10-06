import os
from os.path import join 
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

	collect_plots(MEASURE_FOLDER, os.path.join(MEASURE_FOLDER, 'collected_plots'), 'deloadPhase.png')
 
	#*-----------------------
	#* SINGLE RUN
 	#*----------------------- 
  
	MEASURE_FOLDER = os.path.join(EXP_FOLDER, 'testPArun9')
	df = get_data_frame(MEASURE_FOLDER)
	df.drop(columns=['betaPAErr'], inplace=True)
	df.dropna(inplace=True)
	#freqs = plot_results(df, 384201., save_folder=MEASURE_FOLDER)
 
	max_freq = 384182.5
	data = df
 
	freqs = ((max_freq-PUMP_FREQUENCY)-(data['tempV']-data['tempV'].min())*FREQVSVOLT- (data['currV']-df['currV'].min())*FREQVSCURR)
	fig, ax = plt.subplots()
	plot_spline_fit(ax=ax, x=freqs, y=data['ratio'], yerr=data['ratioErr'], 
				 s=0.0, save_folder=MEASURE_FOLDER, 
				 mfc='red', color='black', 
				 title='')
	plt.show()
	plt.close()
 
	betaPAs = [a for a,b in sorted(zip(data['betaPA'], freqs), key=lambda pair:pair[1])]
	freqs = sorted(freqs)
	plt.plot(freqs, betaPAs, 'o-', ms=5, label="")
	plt.legend()
	plt.xlabel(r'$\Delta$ (GHz)')
	plt.ylabel(r'$\beta_{\mathrm{eff}}$ ')
	#plt.savefig(join(MEASURE_FOLDER, 'betaVsFreq.png'), dpi=200) 
	plt.title(f"2-body Decay Plot {''} ", **titledict) 
	plt.show()
	plt.close()
 
 	# *-----------------------
	# * MULTIPLE RUN COMPARISON
 	# *-----------------------
	names = ['wide1', 'medium1', 'small1', 'smallest1']
	folders = [os.path.join(EXP_FOLDER,'RepumpTrapDepthEffect', path) for path in names ]
	dfs = [get_data_frame(measure_folder) for measure_folder in folders]

	labels = names

	max_freqs = [384182.6]*len(dfs)
	zipped_data = list(zip(dfs, max_freqs))
	fig, ax = plt.subplots()
	for i, (df, max_freq)  in enumerate(zipped_data[:]):
		data = df.dropna()
		freqs = ((max_freq-PUMP_FREQUENCY)-(data['tempV']-data['tempV'].min())*FREQVSVOLT- (data['currV']-data['currV'].min())*FREQVSCURR)
		ax=plot_spline_fit(ax, x=freqs, y=data['ratio'], yerr=data['ratioErr'], scolor=f'C{i}', mfc=f'C{i}',color=f'C{i}', s=0.00, ms=5, label=labels[i])
	plt.legend()
 
	fig, ax = plt.subplots()
	for i, (df, max_freq)  in enumerate(zipped_data[:]):
		data = df.dropna()
		freqs = ((max_freq-PUMP_FREQUENCY)-(data['tempV']-data['tempV'].min())*FREQVSVOLT- (data['currV']-data['currV'].min())*FREQVSCURR)
		betaPAs = [a for a,b in sorted(zip(data['betaPA'], freqs), key=lambda pair:pair[1])]
		freqs = sorted(freqs)
		plt.plot(freqs, betaPAs, 'o-', ms=5, label=labels[i])
	plt.legend()


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
	#* MEGA_RUN
 	#*-----------------------	
	MEASURE_FOLDER = os.path.join(EXP_FOLDER, 'MegaRuns', 'testPArunMega3')
	df = get_data_frame(MEASURE_FOLDER,
						plot=False,
						cache_all=True)
	df.drop(columns=['betaPAErr'], inplace=True)
	df.dropna(inplace=True)
	df = df[df['ratio']<1.2]

	groupbyKey = 'pump_reference'
	titleKey = 'pump_AOM_freq'

	df_grouped = df.groupby(by=groupbyKey)
	min_ratios = df_grouped['ratio'].min()

	groups = dict(list(df_grouped))
	dfs = [df for df in groups.values()]

	# plotting ratio vs freq
	max_freqs = [384182.5]*len(dfs)
	zipped_data = list(zip(dfs, max_freqs))
	fig, ax = plt.subplots()
	for i, (df, max_freq)  in enumerate(zipped_data[:]):
		data = df
		freqs = ((max_freq-PUMP_FREQUENCY)-(data['tempV']-data['tempV'].min())*FREQVSVOLT- (data['currV']-data['currV'].min())*FREQVSCURR)
		ax=plot_spline_fit(ax, x=freqs, y=data['ratio'], yerr=data['ratioErr'],scolor=f'C{i}', mfc=f'C{i}',color=f'C{i}', s=0.0, ms=5, figsize=(10, 10), label=f"{groupbyKey} = { data.iloc[10][groupbyKey] :.2f}", linewidth=2.5)

	plt.legend()
	plt.savefig(os.path.join(MEASURE_FOLDER, 'lossFeatures.png'))
	plt.title(f'Loss Features, {titleKey} = {data[titleKey].mean():.2f}', **titledict)
	plt.show()
	plt.close()
	#---------------------------------------------------
	# x = [df[groupbyKey].mean() for df in dfs]
	# y = [(df['ratio'].max() - df['ratio'].min()) for df in dfs]
	# plt.plot( x, y ,'-o')
	# plt.xlabel(groupbyKey)
	# plt.ylabel(r'SNR $ = V_{ss, off} - V_{ss, on}$ ')
	# plt.title(f'SNR Plot, {titleKey} = {df[titleKey].mean():.2f}', **titledict)
	# plt.show()
	# plt.savefig(join(MEASURE_FOLDER, 'SNRplot.png'), dpi=200)
	# plt.close()

	#---------------------------------------------------

	for i, df  in enumerate(dfs[:]):
		data = df
		freqs = ((max_freqs[0]-PUMP_FREQUENCY)-(data['tempV']-data['tempV'].min())*FREQVSVOLT- (data['currV']-data['currV'].min())*FREQVSCURR)
		betaPAs = [a for a,b in sorted(zip(data['betaPA'], freqs), key=lambda pair:pair[1])]
		freqs = sorted(freqs)
		plt.plot(freqs, betaPAs, 'o-', ms=5, label=f"{groupbyKey}={data.iloc[10][groupbyKey]:.2f}")
	plt.legend()
	plt.xlabel(r'$\Delta$ (GHz)')
	plt.ylabel(r'$\beta_{\mathrm{eff}}$ ')
	plt.savefig(join(MEASURE_FOLDER, 'betaVsFreq.png'), dpi=200) 
	plt.title(f'2-body Decay Plot, {titleKey} = {df[titleKey].mean():.2f}', **titledict) 
	plt.show()
	plt.close()
	#*-----------------------
	#* MULTIPLE MEGARUN
	#*-----------------------
 
	folders = [os.path.join(EXP_FOLDER, 'MegaRuns', path ) for path in ['testPArunMega7', 'testPArunMega8']]

	dfs_mega = [get_data_frame(measure_folder, cache_all=True) for measure_folder in folders]

	groupbyKey = 'pump_reference'
	titleKey = 'pump_AOM_freq'

	dfs_grouped = [df_mega.groupby(by=groupbyKey) for df_mega in dfs_mega]
	min_ratios = [df_grouped['ratio'].min() for df_grouped in dfs_grouped]

	groupss = [dict(list(df_grouped)) for df_grouped in dfs_grouped]
	dfs = [ [df for df in groups.values()] for groups in groupss]
 
	for row in dfs:
		data = row[3]
	
		freqs = ((384182.5-PUMP_FREQUENCY)-(data['tempV']-data['tempV'].min())*FREQVSVOLT- (data['currV']-data['currV'].min())*FREQVSCURR)
		betaPAs = [a for a,b in sorted(zip(data['betaPA'], freqs), key=lambda pair:pair[1])]
		freqs = sorted(freqs)
		plt.plot(freqs, betaPAs, 'o-', ms=5, label=f"{titleKey}={data.iloc[10][titleKey]:.2f}")
		plt.title(f'{groupbyKey} = {data.iloc[10][groupbyKey]:.2f}')
	plt.legend()
	
	#*-----------------------
	#* FULL RUNS
	#*-----------------------
	MEASURE_FOLDER = os.path.join(EXP_FOLDER, 'CATrunOct4')
	df = get_data_frame(MEASURE_FOLDER)
	df.drop(columns='betaPAErr', inplace=True)
	df.dropna(inplace=True)

	dfc = df.copy()
	df_grouped = df.groupby(by=['pump_reference', 'pump_AOM_freq'])
	groups = dict(list(df_grouped))
	dfs = [df for df in groups.values()]

	max_freqs = [384394.7]*len(dfs)
	zipped_data = list(zip(dfs, max_freqs))
	num1 = 3
	fig1, ax1s = plt.subplots(num1)    # detuning
	fig1b, ax1sb = plt.subplots(num1)
	fig1size = (8, num1*6)
	fig1bsize= fig1size
	fig1b.set_size_inches(fig1bsize)

	num2 = 2
	fig2, ax2s = plt.subplots(num2) # pump_reference
	fig2b, ax2sb = plt.subplots(num2)
	fig2size = (8, num2*6)
	fig2bsize= fig2size
	fig2b.set_size_inches(fig2bsize)

	for i, (df, max_freq)  in enumerate(zipped_data[:]):
		j1 = i%num1
		j2 = i//num1
		
		data = df.dropna()
		data = data[data['ratio'] < 1.3]
		freqs = ((max_freq-PUMP_FREQUENCY)-(data['tempV']-data['tempV'].min())*FREQVSVOLT- (data['currV']-0.0)*FREQVSCURR)
		
		pref = data['pump_reference'].mean()
		detuning = 180-2*data['pump_AOM_freq'].mean()
		print(pref, detuning)
		
		
		ax1s[j1] = plot_spline_fit(ax1s[j1], 
								x=freqs, y=data['ratio'], yerr=data['ratioErr']
								,scolor=f'C{j2}', mfc=f'C{j2}',color=f'C{j2}'
								,s=0.0, ms=5,linewidth=1.5, figsize=fig1size
								,label=f"Pump Amplitude = {pref:.2f}", fig=fig1)
		
		ax1s[j1].set_title(f"Detuning  = {detuning:.2f}", **titledict)
		ax1s[j1].legend()
		
		ax2s[j2] = plot_spline_fit(ax2s[j2], 
								x=freqs, y=data['ratio'], yerr=data['ratioErr'],
								scolor=f'C{j1}', mfc=f'C{j1}',color=f'C{j1}', 
								s=0.0, ms=5, linewidth=1.5, figsize=fig2size,
								label=f"Detuning  = {detuning:.2f}", fig=fig2)
		
		ax2s[j2].set_title(f"Pump Amplitude = { pref:.2f}", **titledict)
		ax2s[j2].legend()
		
		
		betaPAs = [a for a,b in sorted(zip(data['betaPA'], freqs), key=lambda pair:pair[1])]
		freqs = sorted(freqs)
		
		ax1sb[j1].plot(freqs, betaPAs, 'o-', color=f'C{j2}', ms=5,  label=f"Pump Amplitude = {pref:.2f}")
		ax1sb[j1].set_xlabel(r'$\Delta$ (GHz)')
		ax1sb[j1].set_ylabel(r'$\beta_{\mathrm{eff}}$ ')
		ax1sb[j1].legend()
		ax1sb[j1].set_title(f"Detuning  = {detuning:.2f}", **titledict)
		
		ax2sb[j2].plot(freqs, betaPAs, 'o-',color=f'C{j1}', ms=5, label=f"Detuning  = {detuning:.2f}")
		ax2sb[j2].set_xlabel(r'$\Delta$ (GHz)')
		ax2sb[j2].set_ylabel(r'$\beta_{\mathrm{eff}}$ ')
		ax2sb[j2].legend()
		ax2sb[j2].set_title(f"Pump Amplitude = { pref:.2f}", **titledict)
		
	fig1.tight_layout()
	fig1b.tight_layout()
	fig2.tight_layout()
	fig2b.tight_layout()

	fig1.savefig(os.path.join(MEASURE_FOLDER, 'lossFeaturesDet.png'))
	plt.show()
	plt.close()
	fig1b.savefig(os.path.join(MEASURE_FOLDER, '2bodyDet.png'))
	plt.show()
	plt.close()

	fig2.savefig(os.path.join(MEASURE_FOLDER, 'lossFeaturesPampl.png'))
	plt.show()
	plt.close()
	fig2b.savefig(os.path.join(MEASURE_FOLDER, '2BodyPampl.png'))
	plt.show()
	plt.close()

	SNRdata = df_grouped['ratio'].max() - df_grouped['ratio'].min()
	SNRdf = SNRdata.reset_index()
	SNRdf.columns = ['pump_reference', 'pump_AOM_freq', 'SNR']
	pivot_table = SNRdf.pivot('pump_reference', 'pump_AOM_freq', 'SNR')
	xticklabels = [f'{180-2*x:.2f}' for x in pivot_table.columns]
	yticklabels = [f'{y:.2f}' for y in pivot_table.index]
	sns.heatmap(pivot_table, annot=True, fmt='.2f', xticklabels=xticklabels, yticklabels=yticklabels)
	plt.xlabel("Detuning (MHz)")
	plt.ylabel("Pump Reference")
	plt.grid()
	plt.savefig(os.path.join(MEASURE_FOLDER, 'heatmap.png'))
	plt.show()
	plt.close()

	# fig, ax = plt.subplots()
	# for i, (df, max_freq)  in enumerate(zipped_data[:]):
	# 	data = df.dropna()
	# 	freqs = ((max_freq-PUMP_FREQUENCY)-(data['tempV']-data['tempV'].min())*FREQVSVOLT- (data['currV']-data['currV'].min())*FREQVSCURR)
		
	# 	ax=plot_spline_fit(ax, x=freqs, y=data['ratio'], yerr=data['ratioErr'],scolor=f'C{i}', mfc=f'C{i}',color=f'C{i}', s=0.0, ms=5, figsize=(10, 10), linewidth=2.5)
		
	# 	plt.title(f"Pump Amplituide = { df.iloc[10]['pump_reference'] :.2f}, \
	# 				sDetuning  = { 180-2*df.iloc[10]['pump_AOM_freq'] :.2f}", **titledict)

	# 	plt.legend()

	# 	plt.savefig(os.path.join(MEASURE_FOLDER, f'lossFeatures{i}.png'))
	# 	plt.show()
		
	# 	plt.close()
	# 	fig, ax = plt.subplots()



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
def save_fit_results(run_path, plot=False, bkfile=False, 
					 CATbaseline=True, MOTbaseline=True,
					 initRFit=True, loadFit=True,deloadFit=True,reloadFit=True,
					 storeFitResults=True):
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
