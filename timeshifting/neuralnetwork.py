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


def calc_model(w, input_arr):
    '''
    Given a set of weights and an input vector, returns the output of 
    the neural network.
    
    Arguments:
        w(list) -- A list of arrays containing model weights from a keras model. 
                   Usually this is output from get_weights()
        input_arr(array) -- An input vector, of the same length as the first 
                            layer of the array representd by w

    Returns:
        array: An output vector.
    '''
    n = len(w)
    if n == 2:
        nodes = np.tanh((np.dot(input_arr, w[0][0])+w[0][1]))
        output = (np.dot(nodes, w[n-1][0])+w[n-1][1])
        return output
    if n == 1:
        output = (np.dot(input_arr, w[n-1][0])+w[n-1][1])
        return output
        
    
    nodes = np.tanh((np.dot(input_arr, w[0][0])+w[0][1]))

    for i in np.arange(1,n-1):
        
        nodes = np.tanh((np.dot(nodes, w[i][0])+w[i][1]))
        
    output = (np.dot(nodes, w[n-1][0])+w[n-1][1])
    
    return output


