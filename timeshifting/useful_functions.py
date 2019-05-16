# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 13:40:03 2017

@author: Taylor
"""

#import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import interpolate    

#pyplot stuff causes problems when run from a terminal, so I deactivated it for now.
#def plot_hist(stuff, n):
#    """
#    Quick and dirty way to plot a 1D histogram, figuring out the bin positions.
#    
#    Parameters:
#    ----------    
#    stuff : array_like
#        Some 1D array of values
#    n: integer
#        Number of bins
#    
#    """
#    import numpy as np
#    hist = np.histogram(stuff[np.isfinite(stuff)], bins = n)
#    width = 1.0 * (hist[1][1] - hist[1][0])
#    center = (hist[1][:-1] + hist[1][1:]) / 2
#    plt.bar(center, hist[0], align='center', width = width)
#    plt.show()


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

#Goes through some text and finds the value associated with some keyword. 
#keywords and values are separated by an = sign, so we have "keyword = value".
def get_keyword(text, keyword):
    '''Search a list of strings  for a keyword, and return what comes after it'''
    for line in text:
        if line.startswith(keyword) and line.find('=') != -1:
            return line[line.find('=')+1:].replace('\n','').strip()
    raise Exception('Keyword \"'+keyword+'\" not found in config.par')

def get_parameter(keyword):
    'Search config.par for a keyword and return the associated string'
    f = open('config.par')
    lines= f.readlines()
    f.close()
    pars = []
    #remove comments
    for line in lines:
        if not line.startswith('#'):
            pars.append(line)
    return get_keyword(pars, keyword)


def deg2rad(theta):
    """ Converts an angle in degrees to radians """
    return theta*np.pi/180.0

def rad2deg(theta):
    """ Converts an angle in radians to degrees """
    return theta*180.0/np.pi

#Print a status bar thing. 
def status(percent):
    """"Outputs a status bar, input is an integer percent"""
    print('\r|'+'='*(percent//4)+' '*(25-percent//4)+'| '+ str(percent)+'%',end='\r')
    sys.stdout.flush()

def status_bar(n, nmax):
    """"Outputs a status bar, input is an integer percent"""
    percent = int(float(n)/float(nmax)*100)
    print('\r|'+'='*(percent//4)+' '*(25-percent//4)+'| '+ str(percent)+'%',end='\r')
    sys.stdout.flush()
    
def keep_finite(q1, q2):
    
    q11 = q1[np.logical_and(np.isfinite(q1), np.isfinite(q2))]
    q22 = q2[np.logical_and(np.isfinite(q1), np.isfinite(q2))]
    return q11, q22
    
    
def gauss(x, *p):
    ''' Calculates a gaussian for a set of x values '''
    A, mu, sigma,c = p
    return np.abs(A)*np.exp(-(x-mu)**2/(2.*sigma**2))+c
    
def unit_vector(vector):
    ''' Returns the unit vector of the vector. '''
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    
def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B