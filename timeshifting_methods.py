# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:32:07 2017

@author: Taylor
"""
import numpy as np
from numpy import linalg as LA
import useful_functions as uf

###############################################################################
#All timeshifting methods
###############################################################################
def flat_shift(*kargs, **parameters):
    """
    Calculate a flat delay time between GOES and ACE. This is done using the 
    ACE and GOES solar wind speed, and position, assuming the solar wind is 
    arranged in planes oriented completely towards the Earth.If the target is 
    not GOES, it will instead calculate the shift to the bowshock.

   """   
   
    #List out the parameters that matter    
    plist= ['target']
    #Add in the defaults
    if not 'target' in parameters:
        parameters['target']  = 'GOES'
    #This lists out the parameter values if asked.
    if len(kargs) == 0 :
        for p in plist:
            print p+' = '+str(parameters[p])
        return 1
            
    
    A1 = kargs[0];A2 = kargs[1];Ab1 = kargs[2];Ab2 = kargs[3];G1 = kargs[4];G2 = kargs[5];ACE_t = kargs[6];ACE = kargs[7];ACE_B_t = kargs[8];ACE_B = kargs[9];GOES_t = kargs[10];GOES = kargs[11]; 
    
    if A1 == -1 or G1 == -1:
        return np.nan    
    
    v = ACE['v'][A1:A2]
    ax = ACE['pos'][A1:A2,0]
    
    if parameters['target'] == 'GOES':
        gx = GOES['pos'][G1:G2,0]
    if parameters['target'] == 'bowshock':
        gx = 13.3*6371.
    else:
        gx = 0.
    
    vel_av = np.nanmean(v)
    delta_x_av = np.nanmean(ax) - np.nanmean(gx)
    
    shift = delta_x_av/vel_av
    
    return shift  
    
def front_angle_shift(*kargs, **parameters):
    """
    Calculate a delay time (s) between GOES and ACE assuming phase fronts oriented 
    at some angle in the xy plane. If the target is not GOES, it will instead 
    calculate the shift to the bowshock.

    
    """ 
    #List out the parameters that matter    
    plist= ['angle' ,'target']
    #Add in the defaults
    if not 'target' in parameters:
        parameters['target']  = 'GOES'
    if not 'angle' in parameters:
        parameters['angle'] = 15.
        
    #This lists out the parameter values if asked.
    if len(kargs) == 0 :
        for p in plist:
            print p+' = '+str(parameters[p])
        return 1
            
    
    A1 = kargs[0];A2 = kargs[1];Ab1 = kargs[2];Ab2 = kargs[3];G1 = kargs[4];G2 = kargs[5];ACE_t = kargs[6];ACE = kargs[7];ACE_B_t = kargs[8];ACE_B = kargs[9];GOES_t = kargs[10];GOES = kargs[11]; 
        
    
    if A1 == -1 or G1 == -1:
        return np.nan
    
    v = ACE['v'][A1:A2]
    ax = ACE['pos'][A1:A2,0]
    ay = ACE['pos'][A1:A2,1]
    
    if parameters['target'] == 'GOES':
        gx = GOES['pos'][G1:G2,0]
        gy = GOES['pos'][G1:G2,1]
    if parameters['target'] == 'bowshock':
        gx = 13.3*6371.
        gy = -0.9 * 6371.
    else:
        gx = 0
        gy = 0
    vel_av = np.nanmean(v)
    delta_x_av = np.nanmean(ax) - np.nanmean(gx)
    delta_y_av = np.nanmean(ay) - np.nanmean(gy)
    
    shift = delta_x_av/vel_av + (delta_y_av*np.tan(uf.deg2rad(parameters['angle']))/vel_av)
    
    return shift     
    
def front_normal_shift(*kargs, **parameters):
    """
    Calculate a delay time (s) between GOES and ACE assuming phase fronts oriented 
    in a direction given by a normal vector.If the target is not GOES, it will instead 
    calculate the shift to the bowshock.
    """
    #List out the parameters that matter    
    plist= ['target', 'normal']
    #Add in the defaults
    if not 'target' in parameters:
        parameters['target']  = 'GOES'
    if not 'normal' in parameters:
        parameters['normal'] = [1,0,0]
        
    #This lists out the parameter values if asked.
    if len(kargs) == 0 :
        for p in plist:
            print p+' = '+str(parameters[p])
        return 1
               
    A1 = kargs[0];A2 = kargs[1];Ab1 = kargs[2];Ab2 = kargs[3];G1 = kargs[4];G2 = kargs[5];ACE_t = kargs[6];ACE = kargs[7];ACE_B_t = kargs[8];ACE_B = kargs[9];GOES_t = kargs[10];GOES = kargs[11]; 
    
    if A1 == -1 or G1 == -1:
        return np.nan       
    n = parameters['normal']
    v = ACE['v'][A1:A2]
    apos = np.mean(ACE['pos'][A1:A2],0)
    
    if parameters['target'] == 'GOES':
        gpos = np.mean(GOES['pos'][G1:G2],0)
    if parameters['target'] == 'bowshock':
        gpos = np.array([13.3*6371.,-0.9*6371.,0.1*6371])
    else:
        gpos = np.array([0.,0.,0.])
    
    n_norm = n/np.linalg.norm(n)
    
    delta_pos = gpos - apos
    vel_av = np.array([-1*np.nanmean(v),0,0])
    
    shift = np.dot(n_norm,delta_pos)/np.dot(n_norm, vel_av)
    
    return shift    
  

def MVAB0_shift(*kargs, **parameters):
    """
    Calculate a solar wind phase front normal vector using the MVAB-0 Method.
    
    In more detail, it calculates a modified covariance matrix for some set of B data, then returns
    the eigenvector corresponding to the smallest eigenvalue. If a good direction canot be found, 
    it returns np.nan.
    
    
    """    
    #List out the parameters that matter    
    plist= ['target', 'ratio']
    #Add in the defaults
    if not 'target' in parameters:
        parameters['target']  = 'GOES'
    if not 'ratio' in parameters:
        parameters['ratio']  = 5
    #This lists out the parameter values if asked.
    if len(kargs) == 0 :
        for p in plist:
            print p+' = '+str(parameters[p])
        return 1
            
    A1 = kargs[0];A2 = kargs[1];Ab1 = kargs[2];Ab2 = kargs[3];G1 = kargs[4];G2 = kargs[5];ACE_t = kargs[6];ACE = kargs[7]; ACE_B_t = kargs[8];ACE_B = kargs[9];GOES_t = kargs[10];GOES = kargs[11]; 
    
    M = np.zeros([3,3])
    P = np.zeros([3,3])
    if Ab1 == -1:
        return np.nan
    B_av = np.array([np.nanmean(ACE_B['B'][Ab1:Ab2,0]), np.nanmean(ACE_B['B'][Ab1:Ab2,1]),np.nanmean(ACE_B['B'][Ab1:Ab2,2])])
    e = B_av/np.sqrt(B_av[0]**2+B_av[1]**2+B_av[2]**2)
    #Create a covariance matrix
    
    for i in range(3):
        for j in range(3):
            M[i,j] = np.nanmean(ACE_B['B'][Ab1:Ab2,i]*ACE_B['B'][Ab1:Ab2,j]) -np.nanmean(ACE_B['B'][Ab1:Ab2,i])*np.nanmean(ACE_B['B'][Ab1:Ab2,j])
            if i == j:
                P[i,j] = 1 - (e[i]*e[j])
            else:
                P[i,j] = -1* (e[i]*e[j])
            if np.isnan(M[i, j]):
                return np.nan
    #Get eigenvalues and eigenvectors
    M = np.dot(np.dot(P,M),P)
    eigenvalues, eigenvectors = LA.eig(M)
    args = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[args]
    eigenvectors = eigenvectors[:,args]
    #The vector corresponding to the middle (absolute value) eigenvalue is the minimum variance direction 

    front_normal = eigenvectors[:,1]
    
    #The x component of the vector should point towards the sun (positive)
    if front_normal[0] < 0:
        front_normal = -1*front_normal
    
    # Do a test. For the result to be valid, the second smallest eigenvalue should be x (5 for now) times larger than the smallest
    e = np.sort(np.abs(eigenvalues))
    
    if eigenvalues[2] / eigenvalues[1] < parameters['ratio']:
        return np.nan
        
    else:
        return front_normal_shift_internal(A1, A2, G1, G2, ACE_t, ACE, GOES_t, GOES, front_normal, **parameters)
  

def MVAB_shift(*kargs, **parameters):    
    """
    Calculate a solar wind phase front normal vector using the Minimum Variance (MVA) Method.
    
    In more detail, it calculates a covariance matrix for some set of B data, then returns
    the direction of least variance. (Basically, the eigenvector corresponding to the
    smallest eigenvalue.) If a good direction canot be found, it returns np.nan.

    """
    
    #List out the parameters that matter    
    plist= ['target', 'ratio']
    #Add in the defaults
    if not 'target' in parameters:
        parameters['target']  = 'GOES'
    if not 'ratio' in parameters:
        parameters['ratio']  = 5
    #This lists out the parameter values if asked.
    if len(kargs) == 0 :
        for p in plist:
            print p+' = '+str(parameters[p])
        return 1
            
    A1 = kargs[0];A2 = kargs[1];Ab1 = kargs[2];Ab2 = kargs[3];G1 = kargs[4];G2 = kargs[5];ACE_t = kargs[6];ACE = kargs[7];ACE_B_t = kargs[8];ACE_B = kargs[9];GOES_t = kargs[10];GOES = kargs[11]; 
    
    if Ab1 == -1:
        return np.nan
    
    M = np.zeros([3,3])
    
    #Create a covariance matrix
    
    for i in range(3):
        for j in range(3):
            M[i,j] = np.nanmean(ACE_B['B'][Ab1:Ab2,i]*ACE_B['B'][Ab1:Ab2,j]) - np.nanmean(ACE_B['B'][Ab1:Ab2,i])*np.nanmean(ACE_B['B'][Ab1:Ab2,j])
            if np.isnan(M[i, j]):
                return np.nan
    #Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = LA.eig(M)
    
    
    #The vector corresponding to the minimum (absolute value) eigenvalue is the minimum variance direction 
    min_e = np.argmin(np.abs(eigenvalues))

    front_normal = eigenvectors[:,min_e]
    
    #The x component of the vector should point towards the sun (positive)
    if front_normal[0] < 0:
        front_normal = -1*front_normal
    
    # Do a test. For the result to be valid, the second smallest eigenvalue should be x (5 for now) times larger than the smallest
    e = np.sort(np.abs(eigenvalues))
    
    if e[1]/e[0] > parameters['ratio']:
        return  front_normal_shift_internal(A1, A2, G1, G2, ACE_t, ACE, GOES_t, GOES, front_normal, **parameters)
    else:
        return np.nan


def cross_product_shift(*kargs, **parameters):    
    """
    Calculate a solar wind phase front normal from the cross product.
    """
    #List out the parameters that matter    
    plist= ['target', 'cutoff_angle', 'dist', 'size']
    #Add in the defaults
    if not 'target' in parameters:
        parameters['target']  = 'GOES'
    if not 'cutoff_angle' in parameters:
        parameters['cutoff_angle']  = 13
    if not 'dist' in parameters:
        parameters['dist']  = 14
    if not 'size' in parameters:
        parameters['size']  = 16       
        
    #This lists out the parameter values if asked.
    if len(kargs) == 0 :
        for p in plist:
            print p+' = '+str(parameters[p])
        return 1
            
    A1 = kargs[0];A2 = kargs[1];Ab1 = kargs[2];Ab2 = kargs[3];G1 = kargs[4];G2 = kargs[5];ACE_t = kargs[6];ACE = kargs[7];ACE_B_t = kargs[8];ACE_B = kargs[9];GOES_t = kargs[10];GOES = kargs[11]; 
        
    
    if Ab1 == -1:
        return np.nan    
  
    B = ACE_B['B'][Ab1:Ab2] 

    center = len(B)/2

    B_b = np.nanmean(B[center - parameters['dist'] -parameters['size']/2: center -parameters['dist']+ parameters['size']/2],0)
    
    B_a = np.nanmean(B[center + parameters['dist'] -parameters['size']/2: center + parameters['dist']+ parameters['size']/2],0)
    #print B_b[0]
    
    if np.isnan(B_b[0]):
        return np.nan
    if np.isnan(B_a[0]):
        return np.nan
    if uf.rad2deg(uf.angle_between(B_b,B_a)) < parameters['cutoff_angle']:
        return np.nan
    
    vec = np.cross(B_b,B_a)

    
    return front_normal_shift_internal(A1, A2, G1, G2, ACE_t, ACE, GOES_t, GOES, vec, **parameters)
        
    
def jackel_shift(*kargs, **parameters):    
    """
    Calculate a solar wind phase front angle from the magnetic field angle.
    
    In more detail, it calculates the angle using the equation 45 deg * sin(2*theta)
    where theta is the angle from the x axis B makes.
    

    """
  #List out the parameters that matter    
    plist= ['target']
    #Add in the defaults
    if not 'target' in parameters:
        parameters['target']  = 'GOES'
    #This lists out the parameter values if asked.
    if len(kargs) == 0 :
        for p in plist:
            print p+' = '+str(parameters[p])
        return 1
            
    
    A1 = kargs[0];A2 = kargs[1];Ab1 = kargs[2];Ab2 = kargs[3];G1 = kargs[4];G2 = kargs[5];ACE_t = kargs[6];ACE = kargs[7];ACE_B_t = kargs[8];ACE_B = kargs[9];GOES_t = kargs[10];GOES = kargs[11]; 
    
    if A1 == -1 or G1 == -1 or Ab1 == -1:
        return np.nan    
      
    Bx = np.nanmean(np.nanmean(ACE_B['B'][A1:A2,0]))
    By = np.nanmean(np.nanmean(ACE_B['B'][A1:A2,1]))

    theta = np.arctan2(By, Bx)
    
    pftheta = -45 * np.sin(2* theta)
    
    return front_angle_shift_internal(A1, A2, G1, G2, ACE_t, ACE, GOES_t, GOES, pftheta, **parameters)
    
    
def empirical_shift (*kargs, **parameters):
    """

   
    """
    
  #List out the parameters that matter    
    plist= ['target']
    #Add in the defaults
    if not 'target' in parameters:
        parameters['target']  = 'GOES'
    #This lists out the parameter values if asked.
    if len(kargs) == 0 :
        for p in plist:
            print p+' = '+str(parameters[p])
        return 1
            
    
    A1 = kargs[0];A2 = kargs[1];Ab1 = kargs[2];Ab2 = kargs[3];G1 = kargs[4];G2 = kargs[5];ACE_t = kargs[6];ACE = kargs[7];ACE_B_t = kargs[8];ACE_B = kargs[9];GOES_t = kargs[10];GOES = kargs[11]; 
    
    if A1 == -1 or G1 == -1 or Ab1 == -1:
        return np.nan       
    
    
    Bx = np.nanmean(np.nanmean(ACE_B['B'][Ab1:Ab2,0]))
    By = np.nanmean(np.nanmean(ACE_B['B'][Ab1:Ab2,1]))

    B_angle = uf.rad2deg(np.arctan2(By, Bx))

    pf_angle = np.nan    

    if B_angle < 45 and B_angle > -45:
        pf_angle = 30
    if B_angle < -135 or B_angle > 135:
        pf_angle = 30
    if B_angle >= -135 and B_angle <= -45:
        pf_angle = 0.6*B_angle+60
    if B_angle >= 45 and B_angle <= 135:
        pf_angle = 0.6*B_angle-48
    
    return front_angle_shift_internal(A1, A2, G1, G2, ACE_t, ACE, GOES_t, GOES, pf_angle, **parameters)
    
def empirical_2_shift (*kargs, **parameters):
    """

   
    """
    
  #List out the parameters that matter    
    plist= ['target', 'constant']
    #Add in the defaults
    if not 'target' in parameters:
        parameters['target']  = 'GOES'
    if not 'constant' in parameters:
        parameters['constant']  = 30
    #This lists out the parameter values if asked.
    if len(kargs) == 0 :
        for p in plist:
            print p+' = '+str(parameters[p])
        return 1
            
    
    A1 = kargs[0];A2 = kargs[1];Ab1 = kargs[2];Ab2 = kargs[3];G1 = kargs[4];G2 = kargs[5];ACE_t = kargs[6];ACE = kargs[7];ACE_B_t = kargs[8];ACE_B = kargs[9];GOES_t = kargs[10];GOES = kargs[11]; 
    
    if A1 == -1 or G1 == -1 or Ab1 == -1:
        return np.nan       
    
    
    Bx = np.nanmean(np.nanmean(ACE_B['B'][Ab1:Ab2,0]))
    By = np.nanmean(np.nanmean(ACE_B['B'][Ab1:Ab2,1]))

    B_angle = uf.rad2deg(np.arctan2(By, Bx))

    pf_angle = np.nan    

    if B_angle < 45 and B_angle > -45:
        pf_angle = parameters['constant']
    if B_angle < -135 or B_angle > 135:
        pf_angle = parameters['constant']
    if B_angle >= -135 and B_angle <= -45:
        pf_angle = 0.6*B_angle+60
    if B_angle >= 45 and B_angle <= 135:
        pf_angle = 0.6*B_angle-48
    
    return front_angle_shift_internal(A1, A2, G1, G2, ACE_t, ACE, GOES_t, GOES, pf_angle, **parameters)





def nn_shift(*kargs, **parameters):
    """

   """   
   
    #List out the parameters that matter    
    plist= ['target']
    #Add in the defaults
    if not 'target' in parameters:
        parameters['target']  = 'GOES'
    #This lists out the parameter values if asked.
    if len(kargs) == 0 :
        for p in plist:
            print p+' = '+str(parameters[p])
        return 1
            
    
    A1 = kargs[0];A2 = kargs[1];Ab1 = kargs[2];Ab2 = kargs[3];G1 = kargs[4];G2 = kargs[5];ACE_t = kargs[6];ACE = kargs[7];ACE_B_t = kargs[8];ACE_B = kargs[9];GOES_t = kargs[10];GOES = kargs[11]; 
    
    if A1 == -1 or G1 == -1:
        return np.nan    
    
    v = np.nanmean(ACE['v'][A1:A2])
    a_pos_x = np.nanmean(ACE['pos'][A1:A2], axis = 0)[0]
    a_pos_y = np.nanmean(ACE['pos'][A1:A2], axis = 0)[1]
    a_pos_z = np.nanmean(ACE['pos'][A1:A2], axis = 0)[2]
    
    a_Bx = np.nanmean(ACE_B['B'][A1:A2], axis = 0)[0]
    a_By = np.nanmean(ACE_B['B'][A1:A2], axis = 0)[1]
    a_Bz = np.nanmean(ACE_B['B'][A1:A2], axis = 0)[2]
    
    
    #print v
    #print a_pos
    #print a_pos/v
    
    a_pos_x = a_pos_x/6376./200
    a_pos_y = a_pos_y/6376./200
    a_pos_z = a_pos_z/6376./200

    a_Bx = a_Bx/5
    a_By = a_By/5
    a_Bz = a_Bz/5
    v = v/400
    
    inp =np.array([v, a_pos_x,a_pos_y,a_pos_z, a_Bx, a_By, a_Bz])
    
    shift = calc_model(parameters['w'], inp)*60*60.
    
    #print shift
    #1/0
    
    return shift  

def ensemble_nn_shift(*kargs, **parameters):
    """

   """   
   
    #List out the parameters that matter    
    plist= ['target']
    #Add in the defaults
    if not 'target' in parameters:
        parameters['target']  = 'GOES'
    #This lists out the parameter values if asked.
    if len(kargs) == 0 :
        for p in plist:
            print p+' = '+str(parameters[p])
        return 1
            
    
    A1 = kargs[0];A2 = kargs[1];Ab1 = kargs[2];Ab2 = kargs[3];G1 = kargs[4];G2 = kargs[5];ACE_t = kargs[6];ACE = kargs[7];ACE_B_t = kargs[8];ACE_B = kargs[9];GOES_t = kargs[10];GOES = kargs[11]; 
    
    if A1 == -1 or G1 == -1:
        return np.nan    
    
    v = np.nanmean(ACE['v'][A1:A2])
    a_pos_x = np.nanmean(ACE['pos'][A1:A2], axis = 0)[0]
    a_pos_y = np.nanmean(ACE['pos'][A1:A2], axis = 0)[1]
    a_pos_z = np.nanmean(ACE['pos'][A1:A2], axis = 0)[2]
    
    a_Bx = np.nanmean(ACE_B['B'][A1:A2], axis = 0)[0]
    a_By = np.nanmean(ACE_B['B'][A1:A2], axis = 0)[1]
    a_Bz = np.nanmean(ACE_B['B'][A1:A2], axis = 0)[2]
    
    
    #print v
    #print a_pos
    #print a_pos/v
    
    a_pos_x = a_pos_x/6376./200
    a_pos_y = a_pos_y/6376./200
    a_pos_z = a_pos_z/6376./200

    a_Bx = a_Bx/5
    a_By = a_By/5
    a_Bz = a_Bz/5
    v = v/400
    
    inp =np.array([v, a_pos_x,a_pos_y,a_pos_z, a_Bx, a_By, a_Bz])
    
    n = len(parameters['w'])
    shift = 0
    for i in range(n):
        shift = shift + (calc_model(parameters['w'][i], inp)*60*60.)
    
    shift = shift/n
    #1/0
    
    return shift  

def const_shift(*kargs, **parameters):
    """

   """   
   
    #List out the parameters that matter    
    plist= ['target']
    #Add in the defaults
    if not 'target' in parameters:
        parameters['target']  = 'GOES'
    #This lists out the parameter values if asked.
    if len(kargs) == 0 :
        for p in plist:
            print p+' = '+str(parameters[p])
        return 1
            
    
    A1 = kargs[0];A2 = kargs[1];Ab1 = kargs[2];Ab2 = kargs[3];G1 = kargs[4];G2 = kargs[5];ACE_t = kargs[6];ACE = kargs[7];ACE_B_t = kargs[8];ACE_B = kargs[9];GOES_t = kargs[10];GOES = kargs[11]; 
    
    
    return 51.*60.  
        
###############################################################################        
#     Functions used by other methods
###############################################################################        
def front_normal_shift_internal(A1, A2, G1, G2, ACE_t, ACE, GOES_t, GOES,n, **parameters):
    """
    Calculate a delay time (s) between GOES and ACE assuming phase fronts oriented 
    in a direction given by a normal vector.If the target is not GOES, it will instead 
    calculate the shift to the bowshock.
    """
    if A1 == -1 or G1 == -1:
        return np.nan
    
    v = ACE['v'][A1:A2]
    apos = np.mean(ACE['pos'][A1:A2],0)
    
    if parameters['target'] == 'GOES':
        gpos = np.mean(GOES['pos'][G1:G2],0)
    if parameters['target'] == 'bowshock':
        gpos = np.array([13.3*6371.,-0.9*6371.,0.1*6371])
    else:
        gpos = np.array([0.,0.,0.])
    
    n_norm = n/np.linalg.norm(n)
    
    delta_pos = gpos - apos
    vel_av = np.array([-1*np.nanmean(v),0,0])
    
    shift = np.dot(n_norm,delta_pos)/np.dot(n_norm, vel_av)
    
    return shift
    
def front_angle_shift_internal(A1, A2, G1, G2, ACE_t, ACE, GOES_t, GOES, angle, **parameters):
    """
    Calculate a delay time (s) between GOES and ACE assuming phase fronts oriented 
    at some angle in the xy plane. If the target is not GOES, it will instead 
    calculate the shift to the bowshock.

    
    """ 
    
    if A1 == -1 or G1 == -1:
        return np.nan
    
    v = ACE['v'][A1:A2]
    ax = ACE['pos'][A1:A2,0]
    ay = ACE['pos'][A1:A2,1]
    
    if parameters['target'] == 'GOES':
        gx = GOES['pos'][G1:G2,0]
        gy = GOES['pos'][G1:G2,1]
    if parameters['target'] == 'bowshock':
        gx = 13.3*6371.
        gy = -0.9 * 6371.
    else:
        gx = 0
        gy = 0
    vel_av = np.nanmean(v)
    delta_x_av = np.nanmean(ax) - np.nanmean(gx)
    delta_y_av = np.nanmean(ay) - np.nanmean(gy)
    
    shift = delta_x_av/vel_av + (delta_y_av*np.tan(uf.deg2rad(angle))/vel_av)
    
    return shift 

def calc_model(w, input_arr):
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