# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:07:35 2019

@author: Taylor
"""

#import timeshifting as ts
import timeshifting.data as data

#Pull Data from CDAWeb
for i in range(2000, 2010):
    data.pull_ACE(i)
    data.pull_ACE_B(i)
    data.pull_GOES(i)
    
for i in range(2000, 2010):
    #Calculates time indices for 2 hour long intervals each separated by half an hour
    data.calc_time_indices(i)
    
for i in range(2000, 2010):
    #Calculates time indices for 2 hour long intervals each separated by half an hour
    data.generate_ideal_timeshifts(i)