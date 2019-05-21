# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:13:47 2019

@author: Taylor
"""
import numpy as np
import timeshifting as ts
import timeshifting.neuralnetwork as nn
import time

start = time.time()

#Timeshifting stuff

#o = False
#
#ts.calc_timeshifts('flat', 'flat', overwrite = o)
#
#ts.calc_timeshifts('MVAB0', 'MVAB0r5', ratio = 5, overwrite = o)
#ts.calc_timeshifts('MVAB0', 'MVAB0r2', ratio = 2, overwrite = o)
#
#ts.calc_timeshifts('MVAB', 'MVABr2', ratio = 2, overwrite = o)
#ts.calc_timeshifts('MVAB', 'MVABr5', ratio = 5, overwrite = o)
#
#ts.calc_timeshifts('front_angle', 'angle_15', angle = 15., overwrite = o)
#
#ts.calc_timeshifts('jackel', 'jackel', overwrite = o)
#
#ts.calc_timeshifts('empirical', 'empirical', overwrite = o)
#
#ts.calc_timeshifts('cross_product', 'cross_product', overwrite = o)
#
#ts.evaluate_method('MVAB0r5', 0.3)
#ts.evaluate_method('MVAB0r2', 0.3)
#
#ts.evaluate_method('MVABr5', 0.3)
#ts.evaluate_method('MVABr2', 0.3)
#
#ts.evaluate_method('flat', 0.3)
#
#ts.evaluate_method('angle_15', 0.3)
#
#ts.evaluate_method('empirical', 0.3)
#
#ts.evaluate_method('cross_product', 0.3)
#
#ts.evaluate_method('jackel', 0.3)

#Neural Network stuff

f = '''
def input_output(self, inds):
    X_data = np.transpose(np.array(
            [-1*self.ACE['v'][inds,0]/400,
             self.ACE['v'][inds,1]/400,
             self.ACE['v'][inds,2]/400,
             self.ACE['pos'][inds,0]/6376./200,
             self.ACE['pos'][inds,1]/6376./200,
             self.ACE['pos'][inds,2]/6376./200,
             self.ACE_B['B'][inds,0]/5.,
             self.ACE_B['B'][inds,1]/5.,
             self.ACE_B['B'][inds,2]/5.,
             ]))
    Y_data = self.ideal_shifts[inds]/60.
    return X_data, Y_data
'''
a = nn.Network(
                 custom_func = f, 
                 filename = 'test2.npy',
                 training_corr_min = 0.7,
                 n_train = 2500,
                 min_shift = 2.,
                 layout = np.array([10,10,10]),
                 n_models = 10,
                 optimizer = 'adam',
                 loss = 'mae',
                 metrics = ['mse'],
                 n_epochs = 100,
                 batch_size = 50,
               )

a.loadData()
a.run()

b = nn.loadNetwork('test2.npy')
b.printSummary()
b.testModelsWidth(corr_min = 0.3)

ts.evaluate_method('flat', corr_min = 0.3)
ts.evaluate_method('MVAB0r2',corr_min = 0.3)

print('')
timetaken = time.time() - start
print('This code took' , timetaken , ' seconds (',timetaken/60.,' mins) to run.')
