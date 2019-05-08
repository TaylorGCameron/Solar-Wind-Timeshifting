# -*- coding: utf-8 -*-
"""
Created on Tue May  7 18:07:35 2019

@author: Taylor
"""
import timeshifting.data as data

#Pull Data from CDAWeb
data.pull_data()
    
#Calculates time indices for 2 hour long intervals each separated by half an hour
data.calc_time_indices()
    
#Calculates time indices for 2 hour long intervals each separated by half an hour
data.generate_ideal_timeshifts()