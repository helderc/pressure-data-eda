# pressure data eda
Pressure Data Exploratory Data Analysis


## Stroke Unit Data: How to properly load it

1. Using the ForeSitePT Analyzer load the raw .xsn file
	1. If it is the fist time you open that file it will ask for saving and processing. In the processing window there will be several parameters, including the "threshold". The default is 5 but I usually put 0. Everything below this threshold is going to be 0 during the processing. Setting it as 0 will give you a more noisy data but I think it is worth if you dont want to loose anything.

2. After the processing, export the data to the .csv format. In the small screen about the frames to be exported, select everything OR, if the file is too big, just the frames of interest. Sometimes I need to select some frames of interest.

3. The .csv can be used directly in my code (function ut.clean_csv(...)) to be cleaned. The output is also a .csv.

4. I also use two other functions to load the 'clean' csv and save it as a pickle file (.pkl), which is shorter and very fast to load:

	`pres_dat = ut.read_csv_data(base_fn + '_csv_clean.csv')`
	
    `ut.data_to_pkl(pres_dat, base_fn + '_pkl.pkl')`
	
5. After that to load the data is as easy as:

	`press_data = ut.load_pkl(base_fn + '_pkl.pkl')`