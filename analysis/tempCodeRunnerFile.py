MEASURE_FOLDER = os.path.join(EXP_FOLDER, 'testPARunMega')
	df = get_data_frame(MEASURE_FOLDER)
	df.dropna(inplace=True)

	df_grouped = df.groupby(by='pump_reference')
	min_ratios = df_grouped['ratio'].min()

	groups = dict(list(df_grouped))
	dfs = [df for df in groups.values()]
 