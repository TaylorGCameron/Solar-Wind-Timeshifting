# -*- coding: utf-8 -*-
"""
Created on Thu May  9 13:21:17 2019

@author: Taylor
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from keras.models import Model
from keras.layers import Dense, Input
import inspect
import os
import timeshifting.useful_functions as uf




class Network(object):
    def __init__(self, 
                 training_corr_min = 0.7,
                 n_train = 2500,
                 min_shift = 10.,
                 layout = np.array([10,10,10]),
                 n_models = 2,
                 optimizer = 'adam',
                 loss = 'mae',
                 metrics = ['mse'],
                 n_epochs = 100, 
                 batch_size = 50,
                 filename = 'test_model.npy',
                 custom_func = ''
                 ):
        

        self.start_year = int(uf.get_parameter('start_year'))
        self.end_year = int(uf.get_parameter('end_year'))
        
        self.custom_func = custom_func
        
        #Parameters for choosing training set
        self.training_corr_min = training_corr_min
        self.n_train = n_train
        self.min_shift = min_shift
        
        #parameters for the networks
        self.layout = layout
        self.n_models = n_models
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        
        
        #training parameters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        #Filepath
        self.filepath = uf.get_parameter('filepath')
        if not os.path.exists(self.filepath+'Models/'):
            os.makedirs(self.filepath+'Models/')
        self.filename = self.filepath+'Models/'+filename
        
        #Internal stuff 
        self.nns = []

    def input_output(self, inds):
        X_data = np.transpose(np.array(
                [-1*self.ACE['v'][inds,0]/400,
                 self.ACE['pos'][inds,0]/6376./200,
                 self.ACE['pos'][inds,1]/6376./200,
                 self.ACE['pos'][inds,2]/6376./200,
                 self.ACE_B['B'][inds,0]/5.,
                 self.ACE_B['B'][inds,1]/5.,
                 self.ACE_B['B'][inds,2]/5.,
                 ]))
        Y_data = self.ideal_shifts[inds]/60.
        return X_data, Y_data
        
    def loadData(self):

        ideals = np.load(self.filepath + 'Ideal_shifts/ideal_shifts_'+str(self.start_year)+'.npy')
        self.ACE = np.load(self.filepath + 'Data/ACE_avg_'+str(self.start_year)+'.npy')
        self.ACE_B = np.load(self.filepath + 'Data/ACE_B_avg_'+str(self.start_year)+'.npy')
        self.GOES = np.load(self.filepath + 'Data/GOES_avg_'+str(self.start_year)+'.npy')

        for year in range(self.start_year+1, self.end_year):
            ideals = np.append(ideals, np.load(self.filepath + 'Ideal_shifts/ideal_shifts_'+str(year)+'.npy'), axis = 0)
            self.ACE = np.append(self.ACE, np.load(self.filepath + 'Data/ACE_avg_'+str(year)+'.npy'), axis = 0)
            self.ACE_B = np.append(self.ACE_B, np.load(self.filepath + 'Data/ACE_B_avg_'+str(year)+'.npy'), axis = 0)
            self.GOES = np.append(self.GOES, np.load(self.filepath + 'Data/GOES_avg_'+str(year)+'.npy'), axis = 0)
    
        self.ideal_shifts = ideals[:,0]
        self.ideal_shifts_corrs = ideals[:,1]
        
        #Get rid of nans? This will need to be changed if I want to utilize all data, instead of just averages.
        self.ACE = self.ACE[np.isfinite(self.ideal_shifts)]
        self.ACE_B = self.ACE_B[np.isfinite(self.ideal_shifts)]
        self.GOES = self.GOES[np.isfinite(self.ideal_shifts)]

        self.ideal_shifts_corrs = self.ideal_shifts_corrs[np.isfinite(self.ideal_shifts)]
        self.ideal_shifts = self.ideal_shifts[np.isfinite(self.ideal_shifts)]
        
        return 1
    
    def run_custom_func(self, f, inds):
        d = {}
        exec(f, globals(), d)
        return d['input_output'](self, inds)
    
    def prepareTrainingData(self):
        self.truth_list = np.logical_and(np.logical_and(self.GOES['pos'][:,0] > 0, self.ideal_shifts_corrs > self.corr_min),self.ideal_shifts > self.min_shift )    
        self.inds = np.random.choice(np.arange(len(self.ideal_shifts))[self.truth_list], size = self.n_train, replace = False)
            
        if self.custom_func == '':
            
            self.X_train, self.Y_train = self.input_output(self.inds)
            self.io_text = inspect.getsourcelines(self.input_output)
        else: 
            self.X_train, self.Y_train = self.run_custom_func(self.custom_func, self.inds)
            self.io_text = self.custom_func
        
    def createNetworks(self):
        for i in range(self.n_models):
            X = Input(shape=(self.X_train.shape[1],), name='X_1')

            if len(self.layout) == 1:
                hidden = Dense(self.layout[0], activation='tanh', name='H0')(X)
                
            else:
                hidden = Dense(self.layout[0], activation='tanh', name='H0')(X)
                for j in range(1, len(self.layout)):
                    hidden = Dense(self.layout[j], activation='tanh', name='H'+str(j))(hidden)

            pred = Dense(1, activation='linear')(hidden)

            model = Model(X, pred)
            model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
            self.nns.append(model)
            
    def trainNetworks(self):  
            self.nns_w = []
            for i in range(self.n_models):
                self.nns[i].fit(self.X_train,
                          self.Y_train,
                          epochs=self.n_epochs,
                          batch_size=self.batch_size)
    
                self.nns_w.append(get_weights(self.nns[i]))
        #print(i)

    def run(self):
        self.loadData()
        self.prepareTrainingData()
        self.createNetworks()
        self.trainNetworks()
        self.saveModel()
    def saveNetworks(self):
            for i in range(len(self.nns)):
                get_weights(self.nns[i], save = self.path+'model_'+str(i).zfill(3))
        #print(i)
    def clearKerasModels(self):
        self.nns = 0
    def clearData(self):
        self.ideal_shifts = [0]
        self.ideal_shifts_corrs = [0]
    
        self.ACE = [0]
        self.ACE_B = [0]
        self.GOES = [0]

        
    def clear(self):
        #clear the keras stuff
        self.clearKerasModels()
        #Clear the SC data
        self.clearData()
        self.X_train = 1
        self.Y_train = 1
        
    def saveModel(self):
        self.clear()
        np.save(self.filename, self)
        
    #Gotta clean up all the warnings thatoccur when I run this.
    def testModelsWidth(self, corr_min = 0.3):
        if len(self.ideal_shifts) == 1:
            self.loadData()
            
        def gauss(x, *p):
            ''' Calculates a gaussian for a set of x values '''
            A, mu, sigma,c = p
            return np.abs(A)*np.exp(-(x-mu)**2/(2.*sigma**2))+c    
        #Make inputs
        
        truth_list = self.ideal_shifts_corrs > corr_min
        #truth_list = ideal_shifts_corrs < 0.3
    
        inds = np.arange(len(self.ideal_shifts))[truth_list]
            
        
        
        if self.custom_func == '':
            
            X_data, Y_data = self.input_output(inds)
        else: 
            X_data, Y_data = self.run_custom_func(self.custom_func, inds)
                    
        #So, load in the models. 
            #self.xx_w = X_data
        Y_predict = np.zeros(len(Y_data))
        for i in range(len(self.nns_w)):
            Y_predict = Y_predict + (calc_model(self.nns_w[i], X_data))[:,0]*60
            #print(i)
        
        
        Y_predict = Y_predict / len(self.nns_w)
        
        #self.ts = Y_predict
        
        deltas = (self.ideal_shifts[truth_list] - Y_predict)/60
        #self.Y_predict = Y_predict
        #self.Y_data  = Y_data
        deltas = deltas[np.isfinite(deltas)]
        
        deltas = deltas[deltas < 40]
        deltas = deltas[deltas > -40]    
        hist = np.histogram(deltas, bins = 79)  
        centers = (hist[1][:-1] + hist[1][1:]) / 2.    
        
        #Fit gaussian        
        p0 = [30., 0., 1., 10]        
        coeff, var_matrix = curve_fit(gauss, centers[2:-2], hist[0][2:-2], p0=p0)
        hist_fit_flat = gauss(centers, *coeff)
        width=np.abs(coeff[2])
        center = coeff[1]
                
        plt.plot(centers, hist[0], '-', label = 'Model')
        plt.plot(centers, hist_fit_flat, label = 'Model fit')
        print('')
        print('Model rms: ',np.sqrt(np.mean(deltas**2)))
        print('Width is ', width)
        print('Center is ', center)
        print(' ')
        return width, center

        
    def printSummary(self):
         print('Contains ', self.n_models,' networks')
         print('Each network is arranged as ',np.append(np.insert(self.layout, 0,self.nns_w[0][0][0].shape[0]),[self.nns_w[0][-1][0].shape[1]]))
         print ('The input vector is given by:')
         if self.custom_func == '':
             print(''.join(self.io_text[0]))
         else:
             for line in self.io_text.split('\n'): print(line)
         print ('')
         print('The training set was ', self.n_train ,'samples using a minimum correlation of ', self.corr_min)
         print('This set was trained for ', self.n_epochs, ' epochs with a batch size of ', self.batch_size)
         print ('The optimizer used was ', self.optimizer, ' and the loss function was ', self.loss)
         

def updateNetwork(filename):
    old = loadNetwork(filename)
    try: 
        new = Network(training_corr_min = old.corr_min,
                 n_train = old.n_train,
                 min_shift = old.min_shift,
                 layout = old.layout,
                 n_models = old.n_models,
                 optimizer = old.optimizer,
                 loss = old.loss,
                 metrics = old.metrics,
                 n_epochs = old.n_epochs, 
                 batch_size = old.batch_size,
                 filename = filename,
                 custom_func = old.custom_func)
    except:
        new = Network(training_corr_min = old.training_corr_min,
                 n_train = old.n_train,
                 min_shift = old.min_shift,
                 layout = old.layout,
                 n_models = old.n_models,
                 optimizer = old.optimizer,
                 loss = old.loss,
                 metrics = old.metrics,
                 n_epochs = old.n_epochs, 
                 batch_size = old.batch_size,
                 filename = filename,
                 custom_func = old.custom_func)

    new.nns_w = old.nns_w
    new.saveModel()

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

def loadNetwork(filename):
    filepath = uf.get_parameter('filepath')
    filename = filepath+'Models/'+filename
    return np.load(filename).item()