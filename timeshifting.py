# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:07:51 2016

@author: Taylor
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.dates as mdate
import scipy
import datetime
from scipy.optimize import curve_fit
import timeshifting_methods as tsm
import useful_functions as uf

#Modify this so if creates a directory called shifts, then another inside with the name of the method, THEN starts putting the data there

#Remember overwrite, which defaults to false
def calc_timeshifts(method, name, **parameters):

    filepath = uf.get_parameter('filepath')
    
    #Make the shifts directory if it didn't exist
    if not os.path.exists(filepath+'Shifts/'):
        os.makedirs(filepath+'Shifts/')
    
    #Make the method directory if it didn't exist
    if not os.path.exists(filepath+'Shifts/'+name+'/'):
        os.makedirs(filepath+'Shifts/'+name+'/')
        
    path = filepath+'Shifts/'+name+'/'

    for i in range(2000,2010):
        print('Starting ', i)
        calc_timeshifts_year(i, path+name+'_shifts_'+str(i)+'.npy', method = method+'_shift', **parameters)    

    print('Done!') 

def calc_timeshifts_year(year, filename, method = 'flat', **parameters):                     
    
    start = time.time()

    filepath = uf.get_parameter('filepath')
    
    if not 'overwrite' in parameters:
        parameters['overwrite'] = False
    
    #First, check whether this data file exists already
    if parameters['overwrite'] == False:
        if os.path.exists(filename):
            print('File '+filename+' already exists! Skipping...')
            return 1
        
    ACE = np.load(filepath+'Data/ACE_'+str(year)+'.npy') 
    ACE_B = np.load(filepath+'Data/ACE_B_'+str(year)+'.npy')
    GOES = np.load(filepath+'Data/GOES_'+str(year)+'.npy')
            
    ACE_t = ACE['t'].copy()
    ACE_B_t = ACE_B['t'].copy()
    GOES_t = GOES['t'].copy()
            
                                                          

    A_i = np.load(filepath+'Indices/ACE_indices_'+str(year)+'.npy')
    Ab_i = np.load(filepath+'Indices/ACE_B_indices_'+str(year)+'.npy')
    G_i = np.load(filepath+'Indices/GOES_indices_'+str(year)+'.npy')

    shifts = np.zeros(len(A_i)) + np.nan
      
    if not hasattr(tsm,method):
        print('That method doesnt exist!')
        return -1
    
    timeshifting_method = getattr(tsm, method)
        
    print('Starting '+ method +' method')
    timeshifting_method(**parameters)    

    #Loop through start times
    for i in range(0,len(A_i)):  
        shifts[i] = timeshifting_method(A_i[i,0],A_i[i,1], Ab_i[i,0],Ab_i[i,1],G_i[i,0],G_i[i,1],ACE_t, ACE,ACE_B_t, ACE_B, GOES_t, GOES, **parameters)
        if np.mod(i,200) == 0 and i != 0:
            uf.status(int(float(i)/float(len(A_i))*100))    


    print('')
    timetaken = time.time() - start
    print(timetaken , ' seconds')
    
    
    np.save(filename,  shifts)
    
    
def evaluate_method(method, corr_min, exclude = []):

    filepath = uf.get_parameter('filepath')
        
    #ideal_shifts = np.load('C:/Users/Taylor/Google Drive/Science/Data/timeshifting/ideal_shifts.npy')
    #ideal_shifts_corrs = np.load('C:/Users/Taylor/Google Drive/Science/Data/timeshifting/ideal_shifts_corrs.npy')
    
    ideals = np.zeros([0,2])    
    shifts = np.array([])
    for year in range(2000,2010):

        ideals_year = np.load(filepath + 'Ideal_shifts/ideal_shifts_'+str(year)+'.npy')
        ideals = np.append(ideals, ideals_year, axis = 0)

        year_shifts = np.load(filepath + 'Shifts/'+method+'/'+method+'_shifts_'+str(year)+'.npy')
        shifts = np.append(shifts, year_shifts)
    
    ideal_shifts = ideals[:,0]
    ideal_shifts_corrs = ideals[:,1]
    
    #return ideal_shifts, shifts
    deltas =  (ideal_shifts - shifts)/60.
    
    if exclude != []:
        deltas = np.delete(deltas, exclude)
        ideal_shifts_corrs = np.delete(ideal_shifts_corrs, exclude)

    #Get rid of nans
    ideal_shifts = ideal_shifts[np.isfinite(deltas)]
    ideal_shifts_corrs = ideal_shifts_corrs[np.isfinite(deltas)]
    shifts = shifts[np.isfinite(deltas)]
    deltas = deltas[np.isfinite(deltas)]

    #Get rid of other things
    deltas = deltas[ideal_shifts_corrs > corr_min]
    deltas = deltas[deltas < 40]
    deltas = deltas[deltas > -40]    
    
    hist = np.histogram(deltas, bins = 79)  
    centers = (hist[1][:-1] + hist[1][1:]) / 2.    
    
    #Fit gaussian
    
    p0 = [30., 0., 1., 10]
    
    coeff, var_matrix = curve_fit(uf.gauss, centers[2:-2], hist[0][2:-2], p0=p0)
    hist_fit_flat = uf.gauss(centers, *coeff)
    width=np.abs(coeff[2])
    center = coeff[1]



    plt.plot(centers, hist[0], '-')
    plt.plot(centers, hist_fit_flat)
    print('For ',method,':')
    
    print('Width is ', width)
    print('Center is ', center)
    print('')
    print('')
    return width, deltas

evaluate_method('flat', 0.3)