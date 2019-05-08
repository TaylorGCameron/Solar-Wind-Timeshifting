# Solar Wind Timeshifting


This package contains tools statistical analysis of the performance of different solar wind time-shifting methods. It requires numpy, scipy, matplotlib, and ai.cdas. 

The package requires config.par to be located in the directory the analysis is done in. An example is included.

Before the analysis can be run, data needs to be pulled from CDAWeb and preprocessed. This can be done by running the script "pull_and_process_data.py". This will take hours to run, though it can be interrupted and will skip already generated files when run again. The folder used to store data, along with other key parameters is specified in config.par.

Once preprocessing is finished, to evaluate a timeshift, run timeshifting.calc_timeshifts() for a specific timeshifting method to calculate timeshifts and save them to file. Then, run timeshifting.evaluate() using the folder name specified in calc_timeshifts() to evaluate the error in that method.

All available timeshifting methods are found in ts_methods.py as functions following the naming convention name_shift(). The parameters required for a method can be found by calling the functon for that method without any arguments. New methods can be added by following the style of the other methods. 
