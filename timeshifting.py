# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 13:07:51 2016

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
import timeshifting_methods as tsm
import useful_functions as uf

def load_data(year, GOES_n, mode = 'vm'):
    """
    Given a year and GOES satellite number, return a set of three structures. 
    The first has ACE data for that year, the second ACE B data, and the third 
    GOES data.

    Parameters
    ----------
    year : integer
        Year for which I'm pulling data
    
    GOES_n : integer
        GOES satellite for which I'm pulling data. 
        (Obviously, each GOES satellite only has data for a few years. Below is 
        a list of GOES satellite numbers that work for a each year.)
        2000 - 2003 -> 10, 2004 - 2009 -> 12
        
    mode : string
        Either 'pc', 'mac' or 'vm'. This specifies which kind of machine 
        it's running on, and setting up the correct filepath to the data files.
     """
    if mode == 'vm':
        base = '/data'
    if mode == 'mac':
        base = '/Users/Taylor/Data'
    if mode == 'pc':
        base = 'C:/Users/Taylor/Data'
    
    
    tstart = mdate.date2num(datetime.datetime(year,1,1,1,0,0))
    
    tend = mdate.date2num(datetime.datetime(year,12,31,23,0,0))
    
    #Pull the data

    ACE_data = np.load(base+'/processed/ACE/swepam_'+str(year)+'.npy')
    ACE_tnum = np.load(base+'/processed/ACE/swepam_'+str(year)+'_tnum.npy')

    ACE_Bdata = np.load(base+'/processed/ACE/magswe_'+str(year)+'.npy')
    ACE_Btnum = np.load(base+'/processed/ACE/magswe_'+str(year)+'_tnum.npy')    

    GOES_pos = np.transpose(np.load(base+'/processed/GOES/GOES'+str(GOES_n)+'/pos_gsm.npy')) # This is in km
    GOES_Bz = np.load(base+'/processed/GOES/GOES'+str(GOES_n)+'/Bz.npy') # Units are probably nT? Check this
    GOES_tnum = np.load(base+'/processed/GOES/GOES'+str(GOES_n)+'/tnum.npy')

    #Let's create some structures to store the data.
    
    dtype = np.dtype([('t','f8'), ('pos','3f8'), ('v', 'f8' ), ('n','f8' ), ('p','f8' )])
    ACE = np.ndarray(len(ACE_tnum), dtype = dtype)
    
    dtype = np.dtype([('t','f8'), ('B','3f8' ), ('B_GSE','3f8')])
    ACE_B = np.ndarray(len(ACE_Btnum), dtype = dtype)
    
    dtype = np.dtype([('t','f8'), ('pos','3f8'), ('B','3f8' )])
    GOES = np.ndarray(len(GOES_tnum), dtype = dtype) 

    #Put the data in the structures

    ACE['t'] = ACE_tnum
    ACE['pos'] = np.transpose([ACE_data['pos_gse_x'],ACE_data['pos_gse_y'],ACE_data['pos_gse_z']]) #I'm pretty sure this is in km
    ACE['v'] = ACE_data['proton_speed'] # Units are in km/s
    ACE['n'] = ACE_data['proton_density'] # 1/cc
    ACE['v'][ACE['v'] < 0] = np.nan
    ACE['n'][ACE['n'] < 0] = np.nan
    ACE['p'] = 1.6726*10**(-6) * ACE['n'] * ACE['v']**2 # Units are nPa
    
    ACE_data = []
    ACE_tnum = []
    
    ACE_B['B'] = np.transpose([ACE_Bdata['B_GSE_X'], ACE_Bdata['B_GSE_Y'], ACE_Bdata['B_GSE_Z']])
    ACE_B['B_GSE'] = np.transpose([ACE_Bdata['B_GSE_X'], ACE_Bdata['B_GSE_Y'], ACE_Bdata['B_GSE_Z']])

    ACE_B['t'] = ACE_Btnum
    ACE_B['B'][ACE_B['B']< -9000] = np.nan
    ACE_B['B_GSE'][ACE_B['B_GSE']< -9000] = np.nan

    ACE_Bdata = []
    ACE_Btnum = []
    
    GOES['t'] = GOES_tnum
    GOES['pos'] = np.transpose(GOES_pos)
    GOES['B'][:,2] = GOES_Bz # So, I'll need to fix this so I read in and store all 3 components of B
 
    #I need to only grab that year for GOES.
    
    GOES = GOES[(GOES['t'] >= tstart) & (GOES['t'] < tend)]
    
    return ACE, ACE_B, GOES 


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
