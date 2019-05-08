# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:07:35 2019

@author: Taylor
"""

import data.pull_data as pull
import data.process_data as process

#Pull Data from CDAWeb
for i in range(2000, 2010):
    pull.pull_ACE(i)
    pull.pull_ACE_B(i)
    pull.pull_GOES(i)
    
for i in range(2000, 2010):
    #Calculates time indices for 2 hour long intervals each separated by half an hour
    process.calc_time_indices(i)
    
for i in range(2000, 2010):
    #Calculates time indices for 2 hour long intervals each separated by half an hour
    process.generate_ideal_timeshifts(i)