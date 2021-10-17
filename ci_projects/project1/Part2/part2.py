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
eta = 0.2               # learning rate
epoch = 1               # how many epochs to run?

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
y = np.array([[0], # data set 1
              [1], # data set 2
              [1], # data set 3
              [0], # data set 4
              [0], # data set 5
              [1]  # data set 6
             ])

############################################################################
############################################################################

#Shared weights and bias for the 6 green shared weights
n1_w = np.array([1.73673761, 1.89791391, -2.10677342, -0.14891209, 0.58306155])

#Shared weights and bias for the 6 orange shared weights
n2_w = np.array([-2.25923303, 0.13723954, -0.70121322, -0.62078008, -0.47961976])

#weights for the output neuron
tmp = np.array([1.20973877, -1.07518386, 0.80691921, -0.29078347, -0.22094764, -0.16915604,
                1.10083444, 0.08251052, -0.00437558, -1.72255825, 1.05755642, -2.51791281, -1.91064012])

#putting weights into 13x1 matrix
n3_w = np.zeros((13,1))
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
        v = np.ones((13, 1))
        # group 1 of neurons (the greens!)
        for loop in range(0,6):
            v[loop] = np.dot(X[inx,loop:5+loop], n1_w)  # neuron fires (x as input)
            v[loop] = sigmoid(v[loop])                  # neuron sigmoid
        # group 2 (the oranges!)
        for loop in range(6,12):
            v[loop] = np.dot(X[inx,loop-6:loop-1], n2_w)# neuron fires (x as input)
            v[loop] = sigmoid(v[loop])                  # neuron sigmoid 
        # layer 2   
        oo = np.dot(np.transpose(v), n3_w) # neuron 3 fires, taking neuron 1 and 2 as input
        o = sigmoid(oo) # hey, result of our net!!!
        
        ################################################################
        ################################################################
                
        # error
        err[k] = err[k] + (np.power((y[inx] - o), 2.0))
             
        ################################################################
        ################################################################
                    
        # backprop time folks!!!
        
        # first error calc's
        delta_1 = (-1.0) * (y[inx] - o)
        delta_2 = sigmoid(o,derive=True) # note how I called it, I passed o=sigmoid(oo)
        
        # for updating the last layer
        delta_ow = np.ones((13, 1))
        # format is
        #  delta_index = (input to final neuron) * (Err derivative * Sigmoid derivative)
        for loop in range(13):
            delta_ow[loop] = v[loop]  *  (delta_1*delta_2)        
            
        # for updating the hidden layer neurons (the green set)
        delta_hw1 = np.zeros(5) # array with 5 elements since n1_w is an array w/ 5 elements
        for loop1 in range(0, 5):  #to run through each of the 5 elements in the delta_hw1 array
            for loop2 in range(0, 6): # do first 6 hidden neurons
                tmp = sigmoid(v[loop2], derive=True) # calc its derivative
                #                           input      Sig der   error from output   weight to output neuron
                #keep a running total for what the delta_hw1 should be 
                delta_hw1[loop1] += X[inx,loop1+loop2]  *  tmp  *  ((delta_1*delta_2)   * n3_w[loop2])
            delta_hw1[loop1] = (delta_hw1[loop1] / 6)

        # for updating the hidden layer neurons (the orange set)
        delta_hw2 = np.zeros(5) # array with 5 elements since n2_w is an array w/ 5 elements
        for loop1 in range(0, 5): #to run through each of the 5 elements in the delta_hw2 array
            for loop2 in range(6, 12): # do second 6 hidden neurons
                tmp = sigmoid(v[loop2],derive=True) # calc its derivative
                #                           input      Sig der   error from output   weight to output neuron
                #keep a running total for what the delta_hw1 should be 
                delta_hw2[loop1] += X[inx,loop1+(loop2-6)]  *  tmp  *  ((delta_1*delta_2)   * n3_w[loop2])
            delta_hw2[loop1] = (delta_hw2[loop1] / 6) #divide sum by 6
        
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

fp = open("weights.txt", "w")
fp.write("Weights for hidden layer, first six neurons: \n" + str(n1_w) + "\n\n")
fp.write("Weights for hidden layer, second six neurons: \n" + str(n2_w) + "\n\n")
fp.write("Weights for final, output layer: \n" + str(n3_w) + "\n\n")
fp.close()


# what were the values (just do forward pass)  
for i in range(6): 
    
    # which to sample
    inx = inds[i]
    
    # forward pass
    v = np.ones((13, 1))
    # group 1 of neurons (the greens!)
    for loop in range(0,6):
        v[loop] = np.dot(X[inx,loop:5+loop], n1_w) # neuron fires (x as input)
        v[loop] = sigmoid(v[loop])                # neuron sigmoid
    # group 2 (the oranges!)
    for loop in range(6,12):
        v[loop] = np.dot(X[inx,loop-6:loop-1], n2_w)# neuron fires (x as input)
        v[loop] = sigmoid(v[loop])                  # neuron sigmoid 
    # layer 2   
    oo = np.dot(np.transpose(v), n3_w) # neuron 3 fires, taking neuron 1 and 2 as input
    o = sigmoid(oo) # hey, result of our net!!!
        
    print(str(i) + ": produced: " + str(o) + " wanted " + str(y[i]))

# plot it        
plt.plot(err)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.title('Convergence Plot')
plt.show()