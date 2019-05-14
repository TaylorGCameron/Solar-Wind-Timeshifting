# Solar Wind Timeshifting


This package contains tools statistical analysis of the performance of different solar wind time-shifting methods. Part of this code is what went into the analysis used in Cameron, T., and Jackel, B. (2016), Quantitative evaluation of solar wind time‐shifting methods, Space Weather, 14, 973– 981, doi:10.1002/2016SW001451. It requires numpy, scipy, matplotlib, and ai.cdas. It also contains code for training simple neural networks to time-shift solar wind data. The neural network code requires keras to run as well.

The package requires config.par to be located in the directory the analysis is done in. An example is included.

Before the analysis can be run, data needs to be pulled from CDAWeb and preprocessed. This can be done by running the script "pull_and_process_data.py". This will take hours to run, though it can be interrupted and will skip already generated files when run again. The folder used to store data, along with other key parameters is specified in config.par.


Timeshifting Analysis Instructions:

Once preprocessing is finished, to evaluate a timeshift, run timeshifting.calc_timeshifts() for a specific timeshifting method to calculate timeshifts and save them to file. Then, run timeshifting.evaluate() using the folder name specified in calc_timeshifts() to evaluate the error in that method.

All available timeshifting methods are found in ts_methods.py as functions following the naming convention name_shift(). The parameters required for a method can be found by calling the functon for that method without any arguments. New methods can be added by following the style of the other methods. 
 


Neural Network Instructions:

The neural network code is contained in timeshifting.neuralnetworks. To use it, create a Network object (ie. a = Network()). Then, the network can be trained on a training set by running the run() method (ie. a.run()). THis will have the network load in data, perform the training, and output the resulting network to a file. the printSummmary() method will print outinformation about the model and training parameters, while testModelsWidth() will evaluate the model's effectiveness at timeshifting data in the same way as the methods tested above.

demonstration.py shows examples of both the timeshifting analysis and the neural network code in action.
