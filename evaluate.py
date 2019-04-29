# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 11:00:00 2017

@author: Taylor
"""
import time
import timeshifting as ts
start = time.time()
import numpy as np
o = False

ts.calc_timeshifts('flat', 'flat', overwrite = o)

ts.calc_timeshifts('MVAB0', 'MVAB0r5', ratio = 5, overwrite = o)
ts.calc_timeshifts('MVAB0', 'MVAB0r2', ratio = 2, overwrite = o)

ts.calc_timeshifts('MVAB', 'MVABr2', ratio = 2, overwrite = o)
ts.calc_timeshifts('MVAB', 'MVABr5', ratio = 5, overwrite = o)

ts.calc_timeshifts('front_angle', 'angle_15', angle = 15., overwrite = o)

ts.calc_timeshifts('jackel', 'jackel', overwrite = o)

ts.calc_timeshifts('empirical', 'empirical', overwrite = o)

ts.calc_timeshifts('cross_product', 'cross_product', overwrite = o)

ts.evaluate_method('MVAB0r5', 0.3)
ts.evaluate_method('MVAB0r2', 0.3)

ts.evaluate_method('MVABr5', 0.3)
ts.evaluate_method('MVABr2', 0.3)

ts.evaluate_method('flat', 0.3)

ts.evaluate_method('angle_15', 0.3)

ts.evaluate_method('empirical', 0.3)

ts.evaluate_method('cross_product', 0.3)

ts.evaluate_method('jackel', 0.3)

#w = np.load('C:/Users/Taylor/model.npy')
#e = np.load('C:/Users/Taylor/exclude.npy')
#ts.calc_timeshifts('nn', 'nn', w = w, overwrite = True)

#ts.evaluate_method('nn', 0.3, exclude = e)


#So, load in the model. 
import os
base = 'C:/Users/Taylor/Google Drive/Science/Data/neural_network/batch_101010_c07_n2500_e1000/'
names = os.listdir(base)
t = np.array([file[-3:] for file in names]) == 'npy'
names = np.array(names)[t]    
#model = load_model('C:/Users/Taylor/model.h5')

w = []
for i in range(len(names)):
    w.append(np.load(base + names[i]))


#ts.calc_timeshifts('ensemble_nn', 'ensemble_nn', w = w, overwrite = True)
#e = np.load('C:/Users/Taylor/exclude_07.npy')

#ts.evaluate_method('ensemble_nn', 0.3)

#ts.evaluate_method('flat', 0.3, exclude = e)

print('')
timetaken = time.time() - start
print('This code took' , timetaken , ' seconds (',timetaken/60.,' mins) to run.')
