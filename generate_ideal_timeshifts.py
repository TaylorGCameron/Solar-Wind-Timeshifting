# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:17:03 2019



@author: Taylor
"""
import numpy as np
import useful_functions as uf
import os
import scipy.stats
import scipy
import time



def generate_ideal_timeshifts(year, interval_length = 2./24., dt = 0.5/24.):
    print('Generating ideal shifts for '+str(year) )
    filepath = uf.get_parameter('filepath')
    
    if not os.path.exists(filepath+'Ideal_shifts/'):
        os.makedirs(filepath+'Ideal_shifts/')
    
    #First, check whether this data file exists already. 
    
    if os.path.exists(filepath+'Ideal_shifts/ideal_shifts_'+str(year)+'.npy'):
        print('File '+filepath+'Ideal_shifts/ideal_shifts_'+str(year)+'.npy'+' already exists! Skipping...')
        return 1
    
    #Load data
    ACE = np.load(filepath+'Data/ACE_'+str(year)+'.npy') 
    GOES = np.load(filepath+'Data/GOES_'+str(year)+'.npy')
    ACE_t = ACE['t'].copy()
    GOES_t = GOES['t'].copy()
    
    ACE_i = np.load(filepath+'Indices/ACE_indices_'+str(year)+'.npy')
    GOES_i = np.load(filepath+'Indices/GOES_indices_'+str(year)+'.npy')
    
    #Create an array of start times and end times for each interval       
    start_times = ACE_t[ACE_i][:,0]
    end_times = ACE_t[ACE_i][:,0]
        
    #Define some arrays to hold stuff.
    corrs = np.zeros([len(start_times), 121])+np.nan 
    extra_shifts = np.arange(-60,61,1)* 60.
    shifts = np.zeros(len(start_times)) + np.nan

    ideal_shifts = np.zeros(len(start_times)) + np.nan
    ideal_corrs = np.zeros(len(start_times)) + np.nan
    
    #Keep track of elapsed time                                                                    
    start = time.time()
    
    #Loop through start times
    for i in range(0,len(start_times)):
        #get the interval we are analyzing
        [At1, At2] = ACE_i[i]
        [Gt1, Gt2] = GOES_i[i]
        #Make sure the interval exists
        if np.isnan(At1) or np.isnan(Gt1):
            continue
        #Make sure there are enough ACE data points
        if len(ACE['p'][At1:At2][np.isfinite(ACE['p'][At1:At2])]) < 20:
            continue
        #Make sure there are enough GOES data points
        if len(GOES['B'][Gt1:Gt2,2][np.isfinite(GOES['B'][Gt1:Gt2,2])]) < 20:
            continue 
        #Calculate the flat timeshift as a baseline
        shifts[i] = flat_shift(ACE_i[i], ACE, GOES_i[i], GOES)
        
        #For GOES, supply only a subset of data to save on time. 
        #We are looking at intervals within an hour of the flat timeshift.
        #So maybe supply GOES data from two hours before the interval to 5 after?
        GOES_subset_i = uf.interval(start_times[i]-1./24., end_times[i]+5./24., GOES_t)
        GOES_t_subset = GOES_t[GOES_subset_i[0]: GOES_subset_i[1]]
        GOES_subset = GOES[GOES_subset_i[0]: GOES_subset_i[1]]
        
        #Try shifts within 60 minutes of the flat timeshift
        for j in range(121):
            
            corrs[i, j] = shift_correlate(i, ACE_i, ACE_t, ACE, GOES_t_subset, GOES_subset, shifts[i]+extra_shifts[j])
            
        #Now that we have list of correlations for this interval, we take the highest one, and save it and the corresponding timeshift.
        #Remember to add back the flat timeshift.  

        #If we have nans in the correlation array, abort
        
        if np.isnan(corrs[i,0]):
            ideal_corrs[i] = np.nan
            ideal_shifts[i] = np.nan
            continue

        ideal_corrs[i] = np.nanmax(corrs[i])
        ideal_shifts[i] = shifts[i]+extra_shifts[np.nanargmax(corrs[i])]
        
        #print(shifts[i])
        #print(np.nanargmax(corrs[i]))
        #print(extra_shifts[np.nanargmax(corrs[i])])
        #print('')
        
        #Update a progress bar    
        if np.mod(i,200) == 0 and i != 0:
            uf.status(int(float(i)/float(len(start_times))*100))

    #At the end, save the list of ideal shifts and correlations.
    print('')
    timetaken = time.time() - start
    print(timetaken , ' seconds')
    
    #Package up stuff and save it
    results = np.transpose([ideal_shifts,ideal_corrs])
    np.save(filepath+'Ideal_shifts/ideal_shifts_'+str(year)+'.npy',results)
    
    return 1    

#Given a time interval at ACE, ACE p data and GOES Bz data and a shift time, 
#shift the GOES data back to ACE, reinterpolate the data and compute the correlation between the two data sets.
def shift_correlate(i, ACE_i, ACE_t, ACE, GOES_t, GOES, shift):
    #Make sure the timeshift given is actually a number. If not, return nan.
    if np.isnan(shift):
        return np.nan
    
    #Get times
    t1 = ACE_t[ACE_i[i][0]]
    t2 = ACE_t[ACE_i[i][1]]
    
    #The timeshift needs to be in days to match other times
    shift = (shift)/60./60./24. 

    #Grab ACE data interval indices
    [At1, At2] = ACE_i[i]
    #Grab shifted data interval indices
    [Gt1, Gt2] = uf.interval(t1 + shift, t2 + shift, GOES_t)   
      
    #Check if there is any data
    if np.isnan(At1) or np.isnan(Gt1):
        return np.nan

    #Grab the two values to be correlated
    Bz = GOES['B'][Gt1:Gt2,2]
    p = ACE['p'][At1:At2]
    
    #Make sure there is enough Bz data
    if len(Bz[np.isfinite(Bz)]) < 20:
        return np.nan
    
    GOES_time_forward = GOES_t[Gt1:Gt2] - shift #This is the later time series for GOES, shifted to coincide with ACE

    #Remove linear trend in Bz
    fit = np.polyfit(np.arange(len(Bz[np.isfinite(Bz)])), Bz[np.isfinite(Bz)],1)
    GOES_Bz_shifted = Bz - np.arange(len(Bz))*fit[0] + fit[1]

    #Interpolate the goes data time shifted center around ACE, then resample to ACE time values
    Bz_interpolate = scipy.interpolate.interp1d(GOES_time_forward[np.isfinite(GOES_Bz_shifted)], GOES_Bz_shifted[np.isfinite(GOES_Bz_shifted)], bounds_error = False, fill_value = np.nan)
    Bz_interpolated = Bz_interpolate(ACE_t[At1:At2])

    #So, let's correlate the two data sets that now coincide on the same time values.

    corr = scipy.stats.pearsonr(Bz_interpolated[(np.isfinite(p) & np.isfinite(Bz_interpolated))], p[(np.isfinite(p) & np.isfinite(Bz_interpolated))])[0]

    return corr

def flat_shift(ACE_i, ACE, GOES_i, GOES):

    [At1, At2] = ACE_i
    [Gt1, Gt2] = GOES_i
    
    v = ACE['v'][At1:At2,0]
    ax = ACE['pos'][At1:At2,0]
    
    gx = GOES['pos'][Gt1:Gt2,0]
    
    vel_av = -1*np.nanmean(v)
    delta_x_av = np.nanmean(ax) - np.nanmean(gx)
    
    shift = delta_x_av/vel_av
    
    return shift
    
for i in range(2000, 2010):
    #Calculates time indices for 2 hour long intervals each separated by half an hour
    x = generate_ideal_timeshifts(i)