# Solar Wind Timeshifting


This package runs a statistical analysis of the performance of different solar wind time-shifting methods. It requires numpy, scipy, matplotlib, and ai.cdas. 

To pull down data and perform preprocessing, run the pyrthon script "pull_and_process_data.py". This script runs pulls data from CDAWeb, and does the necessary preprocessing for the rest of the analysis. This will take hours to run, though it can be interrupted and will skip already generated files when run again. The folder used to store data, along with other key parameters is specified in config.par.

Once preprocessing is finished, to evaluate a timeshift, calc_timeshifts() for a specific timeshifting method. Then, run evaluate() using the folder name specified in calc_timeshifts(). Both of these functions are found in timeshifting.py.

All available timeshifting methods are found in timeshifting_methods.py. New methods can be added by following the style of the other methods. 
