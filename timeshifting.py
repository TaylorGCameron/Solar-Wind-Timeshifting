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

def calc_timeshifts_year(year, goes, interval_length, filename, dt = 0.5/24., method = 'flat', **parameters):                     
    
    #interval_length = 2./24.
    
    if not 'overwrite' in parameters:
        parameters['overwrite'] = False
    
    #First, check whether this data file exists already
    if parameters['overwrite'] == False:
        if os.path.exists(filename+'.npy'):
            print('File '+filename+' already exists! Skipping...')
            return 1
        
    ACE, ACE_B, GOES = load_data(year, goes, mode = 'pc')
        
    ACE_t = ACE['t'].copy()
    ACE_B_t = ACE_B['t'].copy()
    GOES_t = GOES['t'].copy()
        
    #Create an array of start times based on year, with each time separated by half an hour
        
    tstart = mdate.date2num(datetime.datetime(year,1,1,1,0,0))
    tend = mdate.date2num(datetime.datetime(year,12,31,23,0,0))
        
    #This is spacing between intervals (in days)
    #dt = 0.5/24.
        
    start_times = np.arange(tstart+3./24.,tend-3./24.,dt)    
    shifts = np.zeros(len(start_times)) + np.nan
                                                          
    #Loop through start times
            
    start = time.time()    
    A_i = np.load('C:/Users/Taylor/Google Drive/Science/Data/timeshifting/indices/'+str(year)+'_ACE_indices.npy')    
    Ab_i = np.load('C:/Users/Taylor/Google Drive/Science/Data/timeshifting/indices/'+str(year)+'_ACE_B_indices.npy')
    G_i = np.load('C:/Users/Taylor/Google Drive/Science/Data/timeshifting/indices/'+str(year)+'_GOES_indices.npy')        
    if not hasattr(tsm,method):
        print('That method doesnt exist!')
        return -1
    timeshifting_method = getattr(tsm, method)
        
    print('Starting '+ method +' method')
    timeshifting_method(**parameters)    
    
    for i in range(0,len(start_times)):  
        shifts[i] = timeshifting_method(A_i[i,0],A_i[i,1], Ab_i[i,0],Ab_i[i,1],G_i[i,0],G_i[i,1],ACE_t, ACE,ACE_B_t, ACE_B, GOES_t, GOES, **parameters)
        if np.mod(i,200) == 0 and i != 0:
            uf.status(int(float(i)/float(len(start_times))*100))    


    print('')
    timetaken = time.time() - start
    print(timetaken , ' seconds')
    
    
    np.save(filename,  shifts)
    
def calc_timeshifts(method, name, **parameters):

    path = 'C:/Users/Taylor/Google Drive/Science/Data/timeshifting/shifts_toGOES/'
    year = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
    GOES = [10,10,10,10,12,12,12,12,12,12]

    for i in range(10):
        print('Starting ', year[i])
        calc_timeshifts_year(year[i], GOES[i], 2./24., path+str(year[i])+'_'+name+'_shifts', dt = 0.5/24., method = method+'_shift', **parameters)    

    print('Done!') 
    
    
def evaluate_method(method, corr_min, exclude = []):
    ideal_shifts = np.load('C:/Users/Taylor/Google Drive/Science/Data/timeshifting/ideal_shifts.npy')
    ideal_shifts_corrs = np.load('C:/Users/Taylor/Google Drive/Science/Data/timeshifting/ideal_shifts_corrs.npy')
    

    years = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
    
    shifts = []    
    
    for year in years:
        year_shifts = np.load('C:/Users/Taylor/Google Drive/Science/Data/timeshifting/shifts_toGOES/'+str(year)+'_'+method+'_shifts.npy')
        shifts = np.append(shifts, year_shifts)
    
    
    deltas =  ideal_shifts - shifts/60.
    
    if exclude != []:
        deltas = np.delete(deltas, exclude)
        ideal_shifts_corrs = np.delete(ideal_shifts_corrs, exclude)
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
