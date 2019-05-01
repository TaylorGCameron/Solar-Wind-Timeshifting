# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:51:22 2019

@author: Taylor
"""

import numpy as np
import matplotlib.dates as mdate
import datetime
#import json # for pretty output
from ai import cdas
import useful_functions as uf
import os

#Pulls a year of ACE data from the server, cleans it up and saves it to a file.
def pull_ACE(year, filepath = 'C:/Users/Taylor/Data/Projects/Solar-Wind-Timeshifting/'):
    
    filename = filepath+'ACE_'+str(year)+'.npy'


    #Check if file already exists
    if os.path.exists(filename+'.npy'):
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

#Take a 1D array, and return an array where every n entries were averaged together.
def collapse_down(arr,n):
    return np.mean(arr[:(len(arr)//n)*n].reshape(-1,n), axis=1)

#Pulls a year of ACE magnetic field data, collapses it down to 64 second cadence, and saves it to a file
def pull_ACE_B(year, filepath = 'C:/Users/Taylor/Data/Projects/Solar-Wind-Timeshifting/'):
    
    filename = filepath+'ACE_B_'+str(year)+'.npy'

    #Check if file already exists
    if os.path.exists(filename+'.npy'):
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


def pull_GOES(year, filepath = 'C:/Users/Taylor/Data/Projects/Solar-Wind-Timeshifting/'):
    
    filename = filepath+'GOES_'+str(year)+'.npy'


    #Check if file already exists
    if os.path.exists(filename+'.npy'):
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
        
        goes_data = cdas.get_data('sp_phys', GOES_names[GOES_dict[year]][0], t1, t2, GOES_names[GOES_dict[year]][1:])
        
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
    
filepath = 'C:/Users/Taylor/Data/Projects/Solar-Wind-Timeshifting/'

for i in range(2000, 2010):
    pull_ACE(i, filepath = filepath)
    pull_ACE_B(i, filepath = filepath)
    pull_GOES(i, filepath = filepath)