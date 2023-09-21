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
SCAT_FOLDER = r"C:\Users\svars\OneDrive\Desktop\UBC Lab\CATExperiment\CATMeasurements\relScatRate"

def dump():
    # *-----------------------
	# *PLOTTING HEATMAP
	# *-----------------------
	pivot_table = scat_df.pivot('pa2', 'pd2', 'depth_ratio')
	sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='magma')
	plt.grid()
 
	# heatmap for scattering
	Is = np.linspace(3, 18, 5)
	deltas = np.array([-6,-8,-10,-12,-14])

	X,Y = np.meshgrid(Is, deltas)
	plt.gcf().set_size_inches(10, 10)

	sns.heatmap(scat_rate(X,Y),  annot=True, fmt='.2f', cmap='magma')
	plt.xticks(range(len(Is)), [f'{I:.2f}' for I in Is])
	plt.yticks(range(len(deltas)), [f'{delta:.2f}' for delta in deltas])

	plt.xlabel('Intensity (mW/cn2)')
	plt.ylabel('Detunings (MHz)')

	plt.grid()

	# 3-D contour plot	
	fig = plt.figure()
	ax = plt.axes(projection='3d')
	ax.contour3D(X, Y, scat_rate(X,Y), 20,  cmap='binary')
	ax.set_xlabel('I')
	ax.set_ylabel('Delta')
	ax.set_zlabel('z')
	ax.set_title('3D contour')
	plt.grid()
	plt.show()

 
def get_timestamp(run_path):
	timestamp = datetime.strptime(os.path.split(run_path)[-1].split('_')[0], dateformat)
	return timestamp

def get_rel_scattering_row(run_path_couple, **kwargs):
	run_path1, run_path2 = run_path_couple
	# run_path1 = os.path.join(data_folder, run_folder1)
	# run_path2 = os.path.join(data_folder, run_folder2)
	result = {}
	with open(os.path.join(run_path1, 'twoSettings.txt'), 'r') as f:
		settings = f.read() 
		import re

		pattern = r'(\w+)\s*=\s*\(([\d.]+),\s*([\d.]+)\)'

		matches = re.findall(pattern, settings)

		
		for match in matches:
			key = match[0]
			value1 = float(match[1])
			value2 = float(match[2])
			result[key] = (value1, value2)

		#print(result)
	with open(os.path.join(run_path2, 'twoSettings.txt'), 'r') as f:
		pass
	
	timestamp = get_timestamp(run_path1)
	fit_results1, fit_results2, fit_results, depth_ratio = get_rel_scattering(run_path1, run_path2, **kwargs)
	
	extracted_values = {
			'pa1': result['pd1'][0],
			'pd1': result['pd1'][1],
			'pa2': result['pd2'][0],
			'pd2': result['pd2'][1],
			'depth_ratio': 1/float(depth_ratio), # ratio between current and std settings
			'timestamp': timestamp,
			**fit_results
		}
	return extracted_values
 
def get_rel_scattering_df(data_folder, parallel=False, **kwargs):
	
	listdir = sorted(os.listdir(data_folder))
	run_path_couple_arr = [[os.path.join(data_folder,run_folder1),os.path.join(data_folder,run_folder2)] \
     for run_folder1, run_folder2 in [listdir[i:i+2] for i in range(len(listdir))[::2]]]
	 
	
	if parallel:
		with Pool(4) as p:
			rows = list(tqdm(p.imap(partial(get_rel_scattering_row, **kwargs), run_path_couple_arr), total=len(run_path_couple_arr)))
	
	depth_ratios_df = pd.DataFrame(rows)
	
	return depth_ratios_df

def get_rel_scattering(run_path1, run_path2, **kwargs):
	resultDict = {}
	
	resultDict1, scat_ratio1 = _load_single_run_rel_scat(run_path1, plot=True, **kwargs)
	resultDict['scat_ratio1'] = scat_ratio1
	resultDict2, scat_ratio2 = _load_single_run_rel_scat(run_path2, plot=True, **kwargs)
	resultDict['scat_ratio2'] = 1/scat_ratio2
	#print(f"scat_ratio 1, (2 inverted) = {scat_ratio1:.3f}, {1/scat_ratio2:.3f}")
	#scat_ratio_avg = (scat_ratio1+1/scat_ratio2)/2
	scat_ratio_avg = scat_ratio1
	resultDict['scat_ratio_avg'] = scat_ratio_avg
 
	Rratio = resultDict1['initMOTR']/resultDict2['initMOTR']
	resultDict['Rratio'] = Rratio
	#print(fr"Rratio = {resultDict1['initMOTR']:.3f} / {resultDict2['initMOTR']:.3f} = {Rratio:.3f}")
	depthRatio = np.sqrt(Rratio/scat_ratio_avg)
	
	return resultDict1, resultDict2, resultDict, depthRatio


def _load_single_run_rel_scat(run_path, plot=False, cache_all=False):
	filename = os.path.join(run_path, 'data.csv')
	settingsname = os.path.join(run_path, 'Settings.txt')
	
	MAT_fit_cache_path = os.path.join(run_path, 'resultDict.txt')
	fit_results = {}
	if not os.path.exists(MAT_fit_cache_path) or not cache_all:
		
		try: 
			dh1 = MTdataHost(SAMPLE_RATE)
			dh1.loadCATdata(fileName=filename, settingsName=settingsname)

			#dh1.tBaseline = 1
			dh1.setOffset()
			dh1.tCATbackground = dh1.offset
			dh1.tLoad = dh1.tCATbackground + 1
			dh1.timeLoad = 30
			dh1.tReload = dh1.tLoad + dh1.timeLoad
			dh1.tDeload = dh1.tReload
			dh1.timeReload = 1
			dh1.timeDeload = 0
			#dh1.tCATbackground = dh1.tReload + dh1.timeReload 
			dh1.tBaseline = dh1.tReload + dh1.timeReload
			#dh1.setCATtiming()
			dh1.setAllCAT(0.002)

			#print('FITTINGGGGGGGGGGGG')
			fit_results = dh1.getResults(run_path, store=True)
			if plot:
				dh1.storeFits(run_path, combined=True, separate=True)
		except Exception as e:
			print(traceback.format_exc())
			print("Fitting ERROR at ", os.path.basename(run_path), '\n')

			with open(MAT_fit_cache_path, 'w') as f:
				f.write(str('MAT fit failed'))
		
	else:
		print("Accessing cached results from :", os.path.basename(run_path))

		fit_results = open(MAT_fit_cache_path, 'r').read()

		if fit_results == 'MAT fit failed':
			print("Failed fit at :", os.path.basename(run_path))
			fit_results = {}

		else:
			fit_results = eval(open(MAT_fit_cache_path, 'r').read())


	
	#scat_ratio = fit_results['motSS']/(fit_results['reloadVolt']+fit_results['baseVolt'] - (fit_results['CATbackgroundVolt']+fit_results['noLightBackground']))
	scat_ratio = (fit_results['motSS']+fit_results['baseVolt']-(fit_results['CATbackgroundVolt']+fit_results['noLightBackground']))/(fit_results['reloadVolt'])
	fit_results['scat_ratio'] = scat_ratio
 	#scat_ratio = dh1.motSS/(dh1.reloadVolt+dh1.baseVolt - (dh1.CATbackgroundVolt+dh1.noLightBackground))
	
	return fit_results, scat_ratio

def scat_rate(I, delta, gamma=6.065, Is=3.576):
	"""_summary_

	Args:
		I (float): intensity (mW/cm^2)
		delta (float): detuning (Mhz)
		gamma (float, optional): linewidth (Mhz) Defaults to 6.065.
		Is (float, optional): Should be one of 3.576(isotropic), 2.503(pi), 1.669 (sigma+-). Defaults to 1.669.

	Returns:
		_type_: gamma: the scattering rate
	"""
	num = gamma*I/Is
	den = 2*(1+I/Is+4*(delta/gamma)**2)

	return num/den