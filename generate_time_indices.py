# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 12:02:20 2016
This is gonna create an interval index list for use in speeding up timeshift calculations
@author: Taylor
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import time
import matplotlib.dates as mdate
import scipy
import scipy.interpolate
import scipy.stats
import datetime
import sys
from numpy import linalg as LA
from scipy.optimize import curve_fit
import solarwind as sw
def calc_time_indices(year, goes, interval_length, dt, mode ='pc'):
    path = 'C:/Users/Taylor/Data/processed/'
    ACE, ACE_B, GOES = sw.load_data(year, goes, mode = mode)
            
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
            [Abt1, Abt2] = sw.interval(start_times[i], end_times[i], ACE_B_t) 
            [At1, At2] = sw.interval(start_times[i], end_times[i], ACE_t)   
            [Gt1, Gt2] = sw.interval(start_times[i], end_times[i], GOES_t)
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
                sw.status(int(float(i)/float(len(start_times))*100))
    #np.save(path+str(year)+'_ACE_indices',  ACE_time_indices)
    #np.save(path+str(year)+'_GOES_indices', GOES_time_indices)
    np.save(path+str(year)+'_ACE_B_indices',  ACE_B_time_indices)    
    return 1
    
    
year = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]
GOES = [10,10,10,10,12,12,12,12,12,12]
for i in range(10):
    print(year[i])
    print('')
    x = calc_time_indices(year[i],GOES[i],2./24., 0.5/24.)