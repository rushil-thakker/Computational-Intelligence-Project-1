############################################################################
############################################################################

#imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
import sys
import random
import time

############################################################################
############################################################################

# parameters
eta = 0.2              # learning rate
epoch = 1           # how many epochs to run?
RandomShuffle = 0      # do we want to randomly shuffle our data?
use_bias = 0           # set to 0 if no bias

############################################################################
############################################################################

# our nonlinear function (and its der)
def sigmoid(x, derive=False): # x is the input, derive is do derivative or not
    if derive: # ok, says calc the deriv?
        return x * (1.0 - x) # note, you might be thinking ( sigmoid(x) * (1 - sigmoid(x)) )
                           # depends on how you call the function
    return ( 1.0 / (1.0 + np.exp(-x)) )

############################################################################
############################################################################

# our data set
if(use_bias == 0):
    # 6 data points 
    X = np.array([
        [0, 0, 0.8, 0.4, 0.4, 0.1, 0, 0, 0],  
        [0, 0.3, 0.3, 0.8, 0.3, 0, 0, 0, 0],  
        [0, 0, 0, 0, 0.3, 0.3, 0.8, 0.3, 0],
        [0, 0, 0, 0, 0, 0.8, 0.4, 0.4, 0.1],
        [0.8, 0.4, 0.4, 0.1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0.3, 0.3, 0.8, 0.3],  
    ]) 
else:
    # augmented with bias term (the last column)
    X = np.array([
        [0, 0, 0.8, 0.4, 0.4, 0.1, 0, 0, 0, 1],  
        [0, 0.3, 0.3, 0.8, 0.3, 0, 0, 0, 0, 1],  
        [0, 0, 0, 0, 0.3, 0.3, 0.8, 0.3, 0, 1],
        [0, 0, 0, 0, 0, 0.8, 0.4, 0.4, 0.1, 1],
        [0.8, 0.4, 0.4, 0.1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0.3, 0.3, 0.8, 0.3, 1],  
    ]) 
    
# labels
y = np.array([[0], # class 1
              [1], # class 2
              [1],
              [0],
              [0],
              [1]
             ])

############################################################################
############################################################################

# initialize weights with random numbers
#   again, these are the numbers I put in your assignment

# do we want to scale the weights? set to 1 to ignore. set > 1 to enable scaling
rnd_scale = 1 # if we are doing (num*weight)+(num*weight)+.... 9 times, can get outside [0,1] quickly (i.e., saturate!)

np.random.seed(666)
n1_w = np.random.rand(6,9+use_bias) / rnd_scale

np.random.seed(333)
n2_w = np.random.rand(6,9+use_bias) / rnd_scale 

np.random.seed(111)
tmp = np.random.rand(12+use_bias) / rnd_scale
n3_w = np.zeros((12+use_bias,1))
n3_w[:,0] = tmp

# print out the values when we started!!!
print("Started with")
print("hidden layer first six neurons: " + str(n1_w))
print("hidden layer second six neurons: " + str(n2_w))
print("final layer weights: " + str(n3_w))

############################################################################
############################################################################

err = np.zeros((epoch,1)) # lets record error so we can make a convergence plot
inds = np.asarray([0,1,2,3,4,5]) # easy (to read) way to deal with shuffle data or not (indices)
for k in range(epoch): 
    
    # init error
    err[k] = 0    
    
    # random shuffle of data each epoch?
    # inds = np.random.permutation(inds)
    
    # doing online, go through each point, one at a time
    for i in range(6): 
        
        # what index?
        inx = inds[i]
        
        ################################################################
        ################################################################
        
        # forward pass
        v = np.ones((12+use_bias, 1))
        # group 1 of neurons (the greens!)
        for loop in range(0,6):
            v[loop] = np.dot(X[inx,:], n1_w[loop,:])  # neuron fires (x as input)
            v[loop] = sigmoid(v[loop])                # neuron sigmoid
        # group 2 (the oranges!)
        for loop in range(6,12):
            v[loop] = np.dot(X[inx,:], n2_w[loop-6,:])  # neuron fires (x as input)
            v[loop] = sigmoid(v[loop])                  # neuron sigmoid 
        # layer 2   
        oo = np.dot(np.transpose(v), n3_w) # neuron 3 fires, taking neuron 1 and 2 as input
        o = sigmoid(oo) # hey, result of our net!!!
        
        ################################################################
        ################################################################
                
        # error
        err[k] = err[k] + ((1.0/2.0) * np.power((y[inx] - o), 2.0))
             
        ################################################################
        ################################################################
                    
        # backprop time folks!!!
        
        # first error calc's
        delta_1 = (-1.0) * (y[inx] - o)
        delta_2 = sigmoid(o,derive=True) # note how I called it, I passed o=sigmoid(oo)
        
        # for updating the last layer
        delta_ow = np.ones((12+use_bias, 1))
        # format is
        #  delta_index = (input to final neuron) * (Err derivative * Sigmoid derivative)
        for loop in range(12+use_bias):
            delta_ow[loop] = v[loop]  *  (delta_1*delta_2)        
            
        # for updating the hidden layer neurons (the green set)
        delta_hw1 = np.ones((6, 9+use_bias)) # 9 inputs, 6 hidden neurons
        for loop1 in range(0,6): # do first 6 hidden neurons
            tmp = sigmoid(v[loop1], derive=True) # calc its deriv
            for loop2 in range(9+use_bias):
                # format
                #                           input      Sig der   error from output   weight to output neuron
                delta_hw1[loop1,loop2] = X[inx,loop2]  *  tmp  *  ((delta_1*delta_2)   * n3_w[loop1])
        # for updating the hidden layer neurons (the orange set)
        delta_hw2 = np.ones((6, 9+use_bias)) # 9 inputs, 6 hidden neurons
        for loop1 in range(6,12): # do second 6 hidden neurons
            tmp = sigmoid(v[loop1],derive=True)
            for loop2 in range(9+use_bias):
                # format
                #                           input      Sig der   error from output   weight to output neuron
                delta_hw2[loop1-6,loop2] = X[inx,loop2]  *  tmp  *  ((delta_1*delta_2)   * n3_w[loop1])
        
        # update time!!!
        n3_w = n3_w - eta * delta_ow  # hidden layer 2        
        n1_w = n1_w - eta * delta_hw1 # hidden layer 1, first six
        n2_w = n2_w - eta * delta_hw2 # hidden layer 1, second six

# print out what we found        
print("Done training")
print("------------------------------------------------------------------------")
print("------------------------------------------------------------------------")
print("hidden layer first six neurons: " + str(n1_w))
print("hidden layer second six neurons: " + str(n2_w))
print("final layer weights: " + str(n3_w))

# plot it        
plt.plot(err)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.title('Convergence Plot')
plt.show()

# what were the values (just do forward pass)  
for i in range(6): 
    
    # which to sample
    inx = inds[i]
    
    # forward pass
    v = np.ones((12+use_bias, 1))
    # group 1
    for loop in range(0,6):
        v[loop] = np.dot(X[inx,:], n1_w[loop,:])  # neuron fires (x as input)
        v[loop] = sigmoid(v[loop])                # neuron sigmoid
    # group 2
    for loop in range(6,12):
        v[loop] = np.dot(X[inx,:], n2_w[loop-6,:])  # neuron fires (x as input)
        v[loop] = sigmoid(v[loop])                  # neuron sigmoid 
    # layer 2
    oo = np.dot(np.transpose(v), n3_w) # neuron 3 fires, taking neuron 1 and 2 as input
    o = sigmoid(oo) # hey, result of our net!!!
        
    print(str(i) + ": produced: " + str(o) + " wanted " + str(y[i]))