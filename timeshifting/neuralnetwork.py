# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:21:17 2019

@author: Taylor
"""
import numpy as np

def get_weights(model, save = ''):
    '''
    This function takes a keras model and outputs the weights in an array
    unconnected to keras. It can also save the weights to file.
    
    Arguments:
        model(object) -- A keras model
    Keyword Arguments:
        save(str) -- A filename to save the weights to
        
    Returns:
        array: A list containing the weights as a series of numpy arrays.
    '''
    
    n_l = len(model.layers)
    
    w = []
    for i in np.arange(1,n_l):
        w.append(model.layers[i].get_weights())

    if save != '':
        np.save(save, w)
    return w


