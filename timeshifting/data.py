# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:51:22 2019

When directly run, pulls ACE swe and mfi data, and GOES data from 2000 to 2009 
from CDAWeb and stores it in files in a location specified in config.par.

@author: Taylor
"""

import numpy as np
import matplotlib.dates as mdate
import datetime
from ai import cdas
import timeshifting.useful_functions as uf
import os
import scipy.stats
import time

def pull_data():
    '''
    Pull a range (specified in config.par) of years of ACE SWE, ACE MFI, and GOES MFI data from CDAWeb, clean it, and store it in a location specified in config.par.
    
    Arguments:
        
    Returns:
        int: Function finished indicator
    '''
    start_year = int(uf.get_parameter('start_year'))
    end_year = int(uf.get_parameter('end_year'))
    for i in range(start_year, end_year):
        pull_ACE_year(i)
        pull_ACE_B_year(i)
        pull_GOES_year(i)
    return 1

def pull_ACE_year(year):
    '''
    Pull a year of ACE SWE data from CDAWeb, clean it, and store it in a location specified in config.par
    
    Arguments:
        year(int) -- The year for which data will be pulled
        
    Returns:
        int: Function finished indicator
    '''
    filepath = uf.get_parameter('filepath')

    #check if there's a folder there, if not, make it
    if not os.path.exists(filepath+'Data/'):
        os.makedirs(filepath+'Data/')
    
    filename = filepath+'Data/ACE_'+str(year)+'.npy'


    #Check if file already exists
    if os.path.exists(filename):
         print('File '+filename+' already exists! Skipping...')
         return 1
    
    #First create empty structures to hold the data
    
    ACE_dtype = np.dtype([('t','f8'), ('pos','3f8'), ('v', '3f8' ), ('n','f8' ), ('p','f8' ), ('spd','f8')])
    ACE = np.ndarray(0, dtype = ACE_dtype)
    
    print('Pulling ACE swe data from '+str(year) )
    uf.status(0)
    
    #Pull the data from CDAWeb in month chunks
    for i in range(1,13):
        t1 = datetime.datetime(year, i,1)
    
        if i+1<13:
            t2 = datetime.datetime(year, i+1,1)
        else:
            t2 = datetime.datetime(year+1, 1,1)
        #print('Pulling '+str(t1)[0:10] + ' - ' + str(t2)[0:10])

        
        swe_data = cdas.get_data('sp_phys', 'AC_H0_SWE',t1, t2, ['Np', 'Vp', 'V_GSE', 'SC_pos_GSE'])
    
        #make temp structure
        ACE_month = np.ndarray(len(swe_data['EPOCH']), dtype = ACE_dtype)
        
        #throw data into structure and clean it up
        ACE_month['t'] = mdate.date2num(swe_data['EPOCH'])
        ACE_month['pos'] = np.transpose([swe_data['ACE_X-GSE'],swe_data['ACE_Y-GSE'],swe_data['ACE_Z-GSE']])
        ACE_month['n'] = swe_data['H_DENSITY']
        ACE_month['v'] = np.transpose([swe_data['VX_(GSE)'],swe_data['VY_(GSE)'],swe_data['VZ_(GSE)']])
        
        #clean up ACE data
        ACE_month['n'][ACE_month['n'] < -10**30 ] = np.nan
        ACE_month['v'][ACE_month['v'] < -10**30] = np.nan
    
        ACE_month['spd'] = np.sqrt(np.sum(ACE_month['v']**2,axis = 1))    
        ACE_month['p'] = 1.6726*10**(-6) * ACE_month['n'] * ACE_month['spd']**2 # Units are nPa
    
        ACE = np.append(ACE,ACE_month)
        uf.status(int((i/12)*100))

    np.save(filename, ACE)
    print(str(year)+' finished!')
    print('File saved to ' + filename)
    return 1

#Take a 1D array, and return an array where every n entries were averaged together.
def collapse_down(arr,n):
    '''Average every n elements of an array (arr), returning a smaller array of length/n'''
    return np.mean(arr[:(len(arr)//n)*n].reshape(-1,n), axis=1)

#Pulls a year of ACE magnetic field data, collapses it down to 64 second cadence, and saves it to a file
def pull_ACE_B_year(year, filepath = ''):
    '''
    Pull a year of ACE MFI data from CDAWeb, clean it, and store it in a location specified in config.par
    
    Arguments:
        year(int) -- The year for which data will be pulled
        
    Returns:
        int: Function finished indicator
    '''
    
    filepath = uf.get_parameter('filepath')

    #check if there's a folder there, if not, make it
    if not os.path.exists(filepath+'Data/'):
        os.makedirs(filepath+'Data/')
    
    filename = filepath+'Data/ACE_B_'+str(year)+'.npy'


    #Check if file already exists
    if os.path.exists(filename):
         print('File '+filename+' already exists! Skipping...')
         return 1
     
    print('Pulling ACE mfi data from '+str(year) )
    uf.status(0)

    ACE_B_dtype = np.dtype([('t','f8'), ('B','3f8' )])
    ACE_B = np.ndarray(0, dtype = ACE_B_dtype)
    
    for i in range(1,13):
                
        t1 = datetime.datetime(year, i,1)
    
        if i+1<13:
            t2 = datetime.datetime(year, i+1,1)
        else:
            t2 = datetime.datetime(year+1, 1,1)
        #print('Pulling '+str(t1)[0:10] + ' - ' + str(t2)[0:10])

        mfi_data = cdas.get_data('sp_phys', 'AC_H0_MFI', t1, t2, ['BGSEc'])

        ACE_B_month = np.ndarray(len(mfi_data['EPOCH'])//4, dtype = ACE_B_dtype)
       
        np.transpose([collapse_down(mfi_data['BX_GSE'], 4),collapse_down(mfi_data['BY_GSE'], 4),collapse_down(mfi_data['BZ_GSE'], 4)])
        
        ACE_B_month['B'] = np.transpose([collapse_down(mfi_data['BX_GSE'], 4),collapse_down(mfi_data['BY_GSE'], 4),collapse_down(mfi_data['BZ_GSE'], 4)])
        ACE_B_month['t'] = collapse_down(mdate.date2num(mfi_data['EPOCH']), 4)
        
        #Clean bad data
        ACE_B_month['B'][ACE_B_month['B'] < -10**30] = np.nan
        
        #append to the full array
        ACE_B = np.append(ACE_B,ACE_B_month)
        uf.status(int((i/12)*100))

    
    np.save(filename, ACE_B)
    print(str(year)+' finished!')
    print('File saved to ' + filename)


def pull_GOES_year(year, filepath = ''):
    '''
    Pull a year of GOES data from CDAWeb, clean it, and store it in a location specified in config.par. 
    
    Which GOES satellite data comes from depends on the year. 2000-2003 pulls GOES 10, 2003-2009 pulls GOES12.
    
    Arguments:
        year(int) -- The year for which data will be pulled
        
    Returns:
        int: Function finished indicator
    '''
    filepath = uf.get_parameter('filepath')

    #check if there's a folder there, if not, make it
    if not os.path.exists(filepath+'Data/'):
        os.makedirs(filepath+'Data/')
    
    filename = filepath+'Data/GOES_'+str(year)+'.npy'


    #Check if file already exists
    if os.path.exists(filename):
         print('File '+filename+' already exists! Skipping...')
         return 1
    
    print('Pulling GOES data from '+str(year) )
    uf.status(0)
    
    GOES_dtype = np.dtype([('t','f8'), ('pos','3f8'), ('B','3f8' )])
    GOES = np.ndarray(0, dtype = GOES_dtype) 
    
    #This maps a given year to a GOES satellite
    GOES_dict = {2000:10, 2001:10, 2002:10, 2003:10, 2004:12, 2005:12, 2006:12, 2007:12, 2008:12, 2009:12}
    #This dict serves to map a goes satellite to it's name in CDAS and it's associated variable names.
    GOES_names = {10:['G0_K0_MAG', 'SC_pos_se', 'B_GSE_c' ], 12:['GOES12_K0_MAG', 'SC_pos_se', 'B_GSE_c']}

    try:
        x = GOES_dict[year]
    except:
        print("Year is not defined yet, try another one.")
        return -1

    #Again, go month by month. 
    for i in range(1,13):
                
        t1 = datetime.datetime(year, i,1)
        
        if i+1<13:
            t2 = datetime.datetime(year, i+1,1)
        else:
            t2 = datetime.datetime(year+1, 1,1)
        #print('Pulling '+str(t1)[0:10] + ' - ' + str(t2)[0:10])
        
        try:
            goes_data = cdas.get_data('sp_phys', GOES_names[GOES_dict[year]][0], t1, t2, GOES_names[GOES_dict[year]][1:])
        except:
            import calendar
            print('No data found for ' + calendar.month_name[i] + ' ' + str(year))
            continue
        GOES_month = np.ndarray(len(goes_data['EPOCH']), dtype = GOES_dtype)
        
        GOES_month['pos'] = np.transpose([goes_data['GSE_X'], goes_data['GSE_Y'], goes_data['GSE_Z'], ])
        GOES_month['B'] = np.transpose([goes_data['BX_GSE'], goes_data['BY_GSE'], goes_data['BZ_GSE'], ])
        GOES_month['t'] = mdate.date2num(goes_data['EPOCH'])
                
        #Clean bad data
        GOES_month['B'][GOES_month['B'] < -10**30] = np.nan
                
        #append to the full array
        GOES = np.append(GOES,GOES_month)
        uf.status(int((i/12)*100))

    
    np.save(filename, GOES)
    print(str(year)+' finished!')
    print('File saved to ' + filename)
    
def calc_time_indices():
    '''
    Create and save to file three lists of indices for each year, for ACE swe, ACE mfi, and GOES.
    The indices define time intervals separated by a time dt, of length interval_length, both contained 
    in config.par. The range of years computed for are also contained in config.par.
    
    Arguments:
        year(int) -- The year for which indices will be calculated
        
    Returns:
        int: Function finished indicator
    '''
    start_year = int(uf.get_parameter('start_year'))
    end_year = int(uf.get_parameter('end_year'))
    for i in range(start_year, end_year):
        calc_time_indices_year(i)
    return 1
    
def calc_time_indices_year(year):
    '''
    Create and save to file three lists of indices for one year, for ACE swe, ACE mfi, and GOES.
    The indices define time intervals separated by a time dt, of length interval_length. 
    (These are defined in config.par)
    
    Arguments:
        year(int) -- The year for which indices will be calculated
        
    Returns:
        int: Function finished indicator
    '''
    
    filepath = uf.get_parameter('filepath')    

    interval_length = eval(uf.get_parameter('interval_length'))
    dt = eval(uf.get_parameter('dt'))

    print('Calculating indices for '+str(year))

    if not os.path.exists(filepath+'Indices/'):
        os.makedirs(filepath+'Indices/')

    filename = filepath+'Indices/ACE_indices_'+str(year)+'.npy'

    #Check if file already exists
    if os.path.exists(filename):
         print('File '+filename+' already exists! Skipping...')
         return 1    
    
    ACE = np.load(filepath+'Data/ACE_'+str(year)+'.npy') 
    ACE_B = np.load(filepath+'Data/ACE_B_'+str(year)+'.npy')
    GOES = np.load(filepath+'Data/GOES_'+str(year)+'.npy')
            
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
            [Abt1, Abt2] = uf.interval(start_times[i], end_times[i], ACE_B_t) 
            [At1, At2] = uf.interval(start_times[i], end_times[i], ACE_t)   
            [Gt1, Gt2] = uf.interval(start_times[i], end_times[i], GOES_t)
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

    np.save(filepath+'/ACE_indices_'+str(year)+'.npy',  ACE_time_indices)
    np.save(filepath+'/GOES_indices_'+str(year)+'.npy', GOES_time_indices)
    np.save(filepath+'/ACE_B_indices_'+str(year)+'.npy',  ACE_B_time_indices)    
    print('')
    return 1
    
def generate_ideal_timeshifts():
    '''
Generate and save a list of ideal (correct) timeshifts generated from cross-correlating
ACE solar wind dynamic pressure and GOES Bz.

    Arguments:
        
    Returns:
        int: Function finished indicator
    '''
    start_year = int(uf.get_parameter('start_year'))
    end_year = int(uf.get_parameter('end_year'))
    for i in range(start_year, end_year):
        generate_ideal_timeshifts_year(i)
    return 1

def generate_ideal_timeshifts_year(year):
    '''
    Generate and save a list of ideal (correct) timeshifts generated from cross-correlating
ACE solar wind dynamic pressure and GOES Bz for one year.

    Arguments:
        year(int) -- The year for which ideal timeshifts will be calculated
        
    Returns:
        int: Function finished indicator
    '''
    print('Generating ideal shifts for '+str(year) )
    filepath = uf.get_parameter('filepath')
    
#   interval_length = eval(uf.get_parameter('interval_length'))
#    dt = eval(uf.get_parameter('dt'))
    
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
        shifts[i] = _flat_shift(ACE_i[i], ACE, GOES_i[i], GOES)
        
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

#
def shift_correlate(i, ACE_i, ACE_t, ACE, GOES_t, GOES, shift):
    '''
        Given a time interval at ACE, ACE p data and GOES Bz data and a shift time, 
        shift the GOES data back to ACE, reinterpolate the data and compute the correlation
        between the two data sets.

    Arguments:
        i(int) -- Interval number to be shifted
        ACE_i(array) -- list of ACE time interval indices
        ACE_t(array) -- list of ACE data time stamps
        ACE(array) -- All ACE data
        GOES_t(array) --list of GOES data time stamps
        GOES(array) -- All GOES data
        shift(array) -- Time GOES data will be shifted by (in seconds)
        
    Returns:
        int: Correlation between ACE dynamic pressure and the shifted GOES Bz data
    '''
    
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

def _flat_shift(ACE_i, ACE, GOES_i, GOES):
    '''
    Calculate the flat timeshift from ACE to GOES for some set of ACE and GOES data.
    The ACE and GOES data provided should be a subset of the whole year.
    
    Arguments:
        ACE_i(array) -- list of ACE time interval indices
        ACE(array) -- ACE data
        GOES_i(array) --list of GOES time interval indices
        GOES(array) -- GOES data
        
    Returns:
        int: Correlation between ACE dynamic pressure and the shifted GOES Bz data
    '''
    [At1, At2] = ACE_i
    [Gt1, Gt2] = GOES_i
    
    v = ACE['v'][At1:At2,0]
    ax = ACE['pos'][At1:At2,0]
    
    gx = GOES['pos'][Gt1:Gt2,0]
    
    vel_av = -1*np.nanmean(v)
    delta_x_av = np.nanmean(ax) - np.nanmean(gx)
    
    shift = delta_x_av/vel_av
    
    return shift

def average_arr(arr, indices):
    '''
    Averages a structured array based on a list of indices indicating
    start and end chunks of data.
    
    Arguments:
        arr(array): A structured array
        indices(array) An array of size nx2, with elements referring to indices of arr
        
    Returns:
        array: A structured array of length n, containing averages of arr.
    '''
    #First create empty structures to hold the data
    avg = np.full(len(indices),np.nan ,dtype = arr.dtype)
    
    uf.status(0)
    
    for i in range(len(avg)):
        if indices[i,0] == -1:
            continue
        for var in arr.dtype.names:
            if np.isnan(arr[var][indices[i,0]:indices[i,1]]).all():
                continue
            avg[var][i]= np.nanmean(arr[var][indices[i,0]:indices[i,1]], axis = 0)
        if np.mod(i,100) == 0:
            uf.status(int(i/len(indices)*100))
    uf.status(100)
    print('')
    
    return avg
def average_data_year(year):
    '''
    For each interval in a year, compute the average of a bunch of different 
    ACE and GOES quantities, and save them to file for use later.Requires data files to have been downloaded.
    
    Arguments:
        year(int): The year for which the averages will be computed
    
    Returns:
        int: Function completed indicator
    '''

    filepath = uf.get_parameter('filepath')

    print('Starting '+str(year))
    
    #ACE
    #check for file
    if os.path.exists(filepath+'Data/ACE_avg_'+str(year)+'.npy') & os.path.exists(filepath+'Data/ACE_B_avg_'+str(year)+'.npy') & os.path.exists(filepath+'Data/GOES_avg_'+str(year)+'.npy'):
        print('File '+'ACE_avg_'+str(year)+'.npy'+' already exists! Skipping...')
        return 1
    
    ACE = np.load(filepath+'Data/ACE_'+str(year)+'.npy')
    ACE_indices = np.load(filepath+'Indices/ACE_indices_'+str(year)+'.npy')
    
    ACE_avg = average_arr(ACE, ACE_indices)
    np.save(filepath+'Data/ACE_avg_'+str(year)+'.npy' ,ACE_avg)

    #ACE_B
    ACE_B = np.load(filepath+'Data/ACE_B_'+str(year)+'.npy')
    ACE_B_indices = np.load(filepath+'Indices/ACE_B_indices_'+str(year)+'.npy')
    
    ACE_B_avg = average_arr(ACE_B, ACE_B_indices)
    np.save(filepath+'Data/ACE_B_avg_'+str(year)+'.npy' ,ACE_B_avg)

    #GOES
    GOES = np.load(filepath+'Data/GOES_'+str(year)+'.npy')
    GOES_indices = np.load(filepath+'Indices/GOES_indices_'+str(year)+'.npy')
    
    GOES_avg = average_arr(GOES, GOES_indices)
    np.save(filepath+'Data/GOES_avg_'+str(year)+'.npy' ,GOES_avg)
    
    return ACE_avg

def average_data():
    '''
    Computes average_data_year() for each year from start_year to end_year as
    specified in config.par. 
    
    Arguments:
    
    Returns:
        int: Function completed indicator
    '''
    start_year = int(uf.get_parameter('start_year'))
    end_year = int(uf.get_parameter('end_year'))
    for i in range(start_year, end_year):
        average_data_year(i)
    return 1