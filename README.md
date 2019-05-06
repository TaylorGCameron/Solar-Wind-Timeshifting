# Solar Wind Timeshifting


This package runs a statistical analysis of the performance of different solar wind time-shifting methods. It requires numpy, scipy, matplotlib, and ai.cdas. 

To pull down data and perform preprocessing, run the bash script process. This script runs pull data.py, generate_time_indices.py, and generate_ideal_timeshifts.py in order. The folder used to store data, along with other key parameters is specified in config.par.

Once preprocessing is finished, to evaluate a timeshift, run calc_timeshifts() for a specific timeshifting method. Then, run evaluate() using the folder name specified in calc_timeshifts(). Both of these functions are found in timeshifting.py.

All available timeshifting methods are found in timeshifting_methods.py. New methods can be added by following the style of the other methods. 
