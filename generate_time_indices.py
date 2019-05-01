# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 12:02:20 2016
This is gonna create an interval index list for use in speeding up timeshift calculations
@author: Taylor
"""

import numpy as np
import matplotlib.dates as mdate
import os
import datetime
import useful_functions as uf

def interval(t1,t2, t):
    """
    Returns the start and end indices for t that contain all times between t1 and t2
    
    Parameters:
    ----------    
    t1 : float
        Start time (days since 0001-01-01 00:00:00 UTC, plus one)
    t2 : float
        End time (same format as above)
    t : array_like
        A list of times in the same format as above.
    """
    t_out =  np.arange(len(t))[(t >= t1) & (t < t2)]
    if len(t_out) == 0:
        return [np.nan, np.nan]
    return [t_out[0],t_out[-1]]

def calc_time_indices(year, interval_length, dt, filepath = ''):
    
    filename = filepath+'ACE_indicies_'+str(year)+'.npy'

    #Check if file already exists
    if os.path.exists(filename):
         print('File '+filename+' already exists! Skipping...')
         return 1    
    
    ACE = np.load(filepath+'ACE_'+str(year)+'.npy') 
    ACE_B = np.load(filepath+'ACE_B_'+str(year)+'.npy')
    GOES = np.load(filepath+'GOES_'+str(year)+'.npy')
            
    ACE_t = ACE['t'].copy()
    ACE_B_t = ACE_B['t'].copy()
    GOES_t = GOES['t'].copy()
            
    #Create an array of start times based on year, with each time separated by half an hour
            
    tstart = mdate.date2num(datetime.datetime(year,1,1,1,0,0))        
    tend = mdate.date2num(datetime.datetime(year,12,31,23,0,0))

            
    start_times = np.arange(tstart+3./24.,tend-3./24.,dt)        
    end_times = start_times + interval_length
    
    ACE_B_time_indices = np.empty([len(start_times),2], dtype = int)            
    ACE_time_indices = np.empty([len(start_times),2], dtype = int)
    GOES_time_indices = np.empty([len(start_times),2], dtype = int)
    
    for i in range(0,len(start_times)):  
            [Abt1, Abt2] = interval(start_times[i], end_times[i], ACE_B_t) 
            [At1, At2] = interval(start_times[i], end_times[i], ACE_t)   
            [Gt1, Gt2] = interval(start_times[i], end_times[i], GOES_t)
            if np.isnan(At1) or np.isnan(Gt1):
               ACE_B_time_indices[i] = [-1,-1] 
               ACE_time_indices[i] = [-1,-1] 
               GOES_time_indices[i] = [-1,-1]  
               continue
            if len(ACE['p'][At1:At2][np.isfinite(ACE['p'][At1:At2])]) < 20:
                ACE_B_time_indices[i] = [-1,-1] 
                ACE_time_indices[i] = [-1,-1]                
                GOES_time_indices[i] = [-1,-1]  
                continue
            if len(GOES['B'][Gt1:Gt2,2][np.isfinite(GOES['B'][Gt1:Gt2,2])]) < 20:
                ACE_B_time_indices[i] = [-1,-1] 
                ACE_time_indices[i] = [-1,-1]
                GOES_time_indices[i] = [-1,-1]  
                continue        
            ACE_time_indices[i] = [At1,At2]
            ACE_B_time_indices[i] = [Abt1,Abt2] 
            GOES_time_indices[i] = [Gt1,Gt2]  
            if np.mod(i,200) == 0 and i != 0:
                uf.status(int(float(i)/float(len(start_times))*100))

    np.save(filepath+'/ACE_indices_'+str(year)+'str(.npy)',  ACE_time_indices)
    np.save(filepath+'/GOES_indices_'+str(year)+'str(.npy)', GOES_time_indices)
    np.save(filepath+'/ACE_B_indices_'+str(year)+'str(.npy)',  ACE_B_time_indices)    
    return 1
    
filepath = uf.get_parameter('filepath')    

for i in range(2000, 2010):
    #Calculates time indices for 2 hour long intervals each separated by half an hour
    x = calc_time_indices(i, 2./24., 0.5/24., filepath = filepath)