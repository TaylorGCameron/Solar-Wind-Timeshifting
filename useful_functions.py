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
    
def angle_between_signed(v1, v2, vn):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    sign = np.dot(vn, np.cross(v1,v2))
    
    return sign*np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    f = interpolate.interp1d(inds[good], A[good],bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    return B