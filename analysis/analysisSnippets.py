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