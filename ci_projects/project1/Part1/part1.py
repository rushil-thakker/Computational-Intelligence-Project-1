# libs
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from IPython.display import clear_output
import sys
import random

# our nonlinear Sigmoid function (including its derivative)
#  note: I let a=1 in the Sigmoid
def sigmoid(x, derive=False): # x is the input, derive is do derivative or not
    if derive: # ok, says calc the deriv?
        return x * (1.0 - x) # note, you might be thinking ( sigmoid(x) * (1 - sigmoid(x)) )
                           # depends on how you call the function
    return ( 1.0 / (1.0 + np.exp(-x)) )

    # define the six training samples
X = np.array([
    [0, 0, 0.8, 0.4, 0.4, 0.1, 0, 0, 0],  # data point 1
    [0, 0.3, 0.3, 0.8, 0.3, 0, 0, 0, 0],  # data point 2
    [0, 0, 0, 0, 0.3, 0.3, 0.8, 0.3, 0],  # data point 3
    [0, 0, 0, 0, 0, 0.8, 0.4, 0.4, 0.1],  # data point 4
    [0.8, 0.4, 0.4, 0.1, 0, 0, 0, 0, 0],  # data point 5
    [0, 0, 0, 0, 0, 0.3, 0.3, 0.8, 0.3],  # data point 6
]) 

# labels of the above training data
y = np.array([[0], 
              [1], 
              [1],
              [0],
              [0],
              [1]
             ])

# weights with random numbers for green
np.random.seed(666)
n1_w = np.random.rand(9, 1)
n2_w = np.random.rand(9, 1)
n3_w = np.random.rand(9, 1)
n4_w = np.random.rand(9, 1)
n5_w = np.random.rand(9, 1)
n6_w = np.random.rand(9, 1)

# weights with random numbers for orange
np.random.seed(333)
n7_w = np.random.rand(9, 1)
n8_w = np.random.rand(9, 1)
n9_w = np.random.rand(9, 1)
n10_w = np.random.rand(9, 1)
n11_w = np.random.rand(9, 1)
n12_w = np.random.rand(9, 1)

# weights with random numbers for ouput
np.random.seed(111)
n13_w = np.random.rand(12, 1)

print('hidden layer 1, neuron 1 weights')
print(n1_w)
print('hidden layer 1, neuron 2 weights')
print(n2_w)
print('hidden layer 1, neuron 3 weights')
print(n3_w)
print('hidden layer 1, neuron 4 weights')
print(n4_w)
print('hidden layer 1, neuron 5 weights')
print(n5_w)
print('hidden layer 1, neuron 6 weights')
print(n6_w)
print('hidden layer 2, neuron 1 weights')
print(n7_w)
print('hidden layer 2, neuron 2 weights')
print(n8_w)
print('hidden layer 2, neuron 3 weights')
print(n9_w)
print('hidden layer 2, neuron 4 weights')
print(n10_w)
print('hidden layer 2, neuron 5 weights')
print(n11_w)
print('hidden layer 2, neuron 6 weights')
print(n12_w)
print('output layer, neuron 1 weights')
print(n13_w)

###############################################
# Epochs
###############################################
eta = 0.2 # learning rate
max_epoch = 1 # how many epochs? (each epoch will run through all 4 data points)
err = np.zeros((max_epoch,1)) # lets record error to plot (get a convergence plot)
inds = np.asarray([0,1,2,3,4,5]) # array of our 4 indices (data point index references)

for k in range(max_epoch): 
    
    # init error
    err[k] = 0    
    
    # random shuffle of data each epoch?
    #inds = np.random.permutation(inds)
    
    # doing online, go through each point, one at a time
    for i in range(6): 
        
        # what index?
        inx = inds[i]
        
        # forward pass
        # layer 1
        v = np.ones((12, 1))
        v[0] = np.dot(X[inx,:], n1_w) # neuron 1 fires (x as input)
        v[0] = sigmoid(v[0])            # neuron 1 sigmoid
        v[1] = np.dot(X[inx,:], n2_w) # neuron 2 fires (x as input)
        v[1] = sigmoid(v[1])            # neuron 2 sigmoid
        v[2] = np.dot(X[inx,:], n3_w) # neuron 3 fires (x as input)
        v[2] = sigmoid(v[2])            # neuron 3 sigmoid
        v[3] = np.dot(X[inx,:], n4_w) # neuron 4 fires (x as input)
        v[3] = sigmoid(v[3])            # neuron 4 sigmoid
        v[4] = np.dot(X[inx,:], n5_w) # neuron 5 fires (x as input)
        v[4] = sigmoid(v[4])            # neuron 5 sigmoid
        v[5] = np.dot(X[inx,:], n6_w) # neuron 6 fires (x as input)
        v[5] = sigmoid(v[5])            # neuron 6 sigmoid
        v[6] = np.dot(X[inx,:], n7_w) # neuron 7 fires (x as input)
        v[6] = sigmoid(v[6])            # neuron 7 sigmoid
        v[7] = np.dot(X[inx,:], n8_w) # neuron 8 fires (x as input)
        v[7] = sigmoid(v[7])             # neuron 8 sigmoid
        v[8] = np.dot(X[inx,:], n9_w) # neuron 9 fires (x as input)
        v[8] = sigmoid(v[8])            # neuron 9 sigmoid
        v[9] = np.dot(X[inx,:], n10_w) # neuron 10 fires (x as input)
        v[9] = sigmoid(v[9])              # neuron 10 sigmoid
        v[10] = np.dot(X[inx,:], n11_w) # neuron 11 fires (x as input)
        v[10] = sigmoid(v[10])               # neuron 11 sigmoid
        v[11] = np.dot(X[inx,:], n12_w) # neuron 12 fires (x as input)
        v[11] = sigmoid(v[11])               # sigmoid 12 fires

        # layer 2
        oo = np.dot(np.transpose(v), n13_w) # neuron 13 fires, taking neuron 1 and 2 as input
        o = sigmoid(oo) # hey, result of our net!!!
        
        # error
        err[k] = err[k] + (np.power((y[inx] - o), 2.0))

        # backprop time folks!!!
        
        # output layer, our delta is (delta_1 * delta_2)
        delta_1 = (-1.0) * (y[inx] - o)
        delta_2 = sigmoid(o,derive=True) # note how I called it, I passed o=sigmoid(oo)
        
        # now, lets prop it back to the weights
        delta_ow = np.ones((12, 1))
        # format is
        #  delta_index = (input to final neuron) * (Err derivative * Sigmoid derivative)
        delta_ow[0]  = v[0]  *  (delta_1*delta_2)
        delta_ow[1]  = v[1]  *  (delta_1*delta_2)
        delta_ow[2]  = v[2]  *  (delta_1*delta_2)
        delta_ow[3]  = v[3]  *  (delta_1*delta_2)
        delta_ow[4]  = v[4]  *  (delta_1*delta_2)
        delta_ow[5]  = v[5]  *  (delta_1*delta_2)
        delta_ow[6]  = v[6]  *  (delta_1*delta_2)
        delta_ow[7]  = v[7]  *  (delta_1*delta_2)
        delta_ow[8]  = v[8]  *  (delta_1*delta_2)
        delta_ow[9]  = v[9]  *  (delta_1*delta_2)
        delta_ow[10] = v[10]  *  (delta_1*delta_2)
        delta_ow[11] = v[11]  *  (delta_1*delta_2)

        # neuron n1
        delta_3 = sigmoid(v[0],derive=True)
        # same, need to prop back to weights
        delta_hw1 = np.ones((9, 1))
        # format
        #              input     this Sig der     error from output   weight to output neuron
        delta_hw1[0] = X[inx,0]  *  delta_3  *  ((delta_1*delta_2)   *n13_w[0])
        delta_hw1[1] = X[inx,1]  *  delta_3  *  ((delta_1*delta_2)   *n13_w[0])
        delta_hw1[2] = X[inx,2]  *  delta_3  *  ((delta_1*delta_2)   *n13_w[0])     
        delta_hw1[3] = X[inx,3]  *  delta_3  *  ((delta_1*delta_2)   *n13_w[0])
        delta_hw1[4] = X[inx,4]  *  delta_3  *  ((delta_1*delta_2)   *n13_w[0])
        delta_hw1[5] = X[inx,5]  *  delta_3  *  ((delta_1*delta_2)   *n13_w[0])  
        delta_hw1[6] = X[inx,6]  *  delta_3  *  ((delta_1*delta_2)   *n13_w[0])
        delta_hw1[7] = X[inx,7]  *  delta_3  *  ((delta_1*delta_2)   *n13_w[0])
        delta_hw1[8] = X[inx,8]  *  delta_3  *  ((delta_1*delta_2)   *n13_w[0])  

        # neuron n2
        delta_4 = sigmoid(v[1],derive=True)
        # same, need to prop back to weights        
        delta_hw2 = np.ones((9, 1))
        delta_hw2[0] = X[inx,0]  *  delta_4  *   ((delta_1*delta_2)   *n13_w[1])
        delta_hw2[1] = X[inx,1]  *  delta_4  *   ((delta_1*delta_2)   *n13_w[1])
        delta_hw2[2] = X[inx,2]  *  delta_4  *   ((delta_1*delta_2)   *n13_w[1])
        delta_hw2[3] = X[inx,3]  *  delta_4  *   ((delta_1*delta_2)   *n13_w[1])
        delta_hw2[4] = X[inx,4]  *  delta_4  *   ((delta_1*delta_2)   *n13_w[1])
        delta_hw2[5] = X[inx,5]  *  delta_4  *   ((delta_1*delta_2)   *n13_w[1])
        delta_hw2[6] = X[inx,6]  *  delta_4  *   ((delta_1*delta_2)   *n13_w[1])
        delta_hw2[7] = X[inx,7]  *  delta_4  *   ((delta_1*delta_2)   *n13_w[1])
        delta_hw2[8] = X[inx,8]  *  delta_4  *   ((delta_1*delta_2)   *n13_w[1])  

        # neuron n3
        delta_5 = sigmoid(v[2],derive=True)
        # same, need to prop back to weights        
        delta_hw3 = np.ones((9, 1))
        delta_hw3[0] = X[inx,0]  *  delta_5  *   ((delta_1*delta_2)   *n13_w[2])
        delta_hw3[1] = X[inx,1]  *  delta_5  *   ((delta_1*delta_2)   *n13_w[2])
        delta_hw3[2] = X[inx,2]  *  delta_5  *   ((delta_1*delta_2)   *n13_w[2])
        delta_hw3[3] = X[inx,3]  *  delta_5  *   ((delta_1*delta_2)   *n13_w[2])
        delta_hw3[4] = X[inx,4]  *  delta_5  *   ((delta_1*delta_2)   *n13_w[2])
        delta_hw3[5] = X[inx,5]  *  delta_5  *   ((delta_1*delta_2)   *n13_w[2])
        delta_hw3[6] = X[inx,6]  *  delta_5  *   ((delta_1*delta_2)   *n13_w[2])
        delta_hw3[7] = X[inx,7]  *  delta_5  *   ((delta_1*delta_2)   *n13_w[2])
        delta_hw3[8] = X[inx,8]  *  delta_5  *   ((delta_1*delta_2)   *n13_w[2])  

        # neuron n4
        delta_6 = sigmoid(v[3],derive=True)
        # same, need to prop back to weights        
        delta_hw4 = np.ones((9, 1))
        delta_hw4[0] = X[inx,0]  *  delta_6  *   ((delta_1*delta_2)   *n13_w[3])
        delta_hw4[1] = X[inx,1]  *  delta_6  *   ((delta_1*delta_2)   *n13_w[3])
        delta_hw4[2] = X[inx,2]  *  delta_6  *   ((delta_1*delta_2)   *n13_w[3])
        delta_hw4[3] = X[inx,3]  *  delta_6  *   ((delta_1*delta_2)   *n13_w[3])
        delta_hw4[4] = X[inx,4]  *  delta_6  *   ((delta_1*delta_2)   *n13_w[3])
        delta_hw4[5] = X[inx,5]  *  delta_6  *   ((delta_1*delta_2)   *n13_w[3])
        delta_hw4[6] = X[inx,6]  *  delta_6  *   ((delta_1*delta_2)   *n13_w[3])
        delta_hw4[7] = X[inx,7]  *  delta_6  *   ((delta_1*delta_2)   *n13_w[3])
        delta_hw4[8] = X[inx,8]  *  delta_6  *   ((delta_1*delta_2)   *n13_w[3])  

        # neuron n5
        delta_7 = sigmoid(v[4],derive=True)
        # same, need to prop back to weights        
        delta_hw5 = np.ones((9, 1))
        delta_hw5[0] = X[inx,0]  *  delta_7  *   ((delta_1*delta_2)   *n13_w[4])
        delta_hw5[1] = X[inx,1]  *  delta_7  *   ((delta_1*delta_2)   *n13_w[4])
        delta_hw5[2] = X[inx,2]  *  delta_7  *   ((delta_1*delta_2)   *n13_w[4])
        delta_hw5[3] = X[inx,3]  *  delta_7  *   ((delta_1*delta_2)   *n13_w[4])
        delta_hw5[4] = X[inx,4]  *  delta_7  *   ((delta_1*delta_2)   *n13_w[4])
        delta_hw5[5] = X[inx,5]  *  delta_7  *   ((delta_1*delta_2)   *n13_w[4])
        delta_hw5[6] = X[inx,6]  *  delta_7  *   ((delta_1*delta_2)   *n13_w[4])
        delta_hw5[7] = X[inx,7]  *  delta_7  *   ((delta_1*delta_2)   *n13_w[4])
        delta_hw5[8] = X[inx,8]  *  delta_7  *   ((delta_1*delta_2)   *n13_w[4])  

        # neuron n6
        delta_8 = sigmoid(v[5],derive=True)
        # same, need to prop back to weights        
        delta_hw6 = np.ones((9, 1))
        delta_hw6[0] = X[inx,0]  *  delta_8  *   ((delta_1*delta_2)   *n13_w[5])
        delta_hw6[1] = X[inx,1]  *  delta_8  *   ((delta_1*delta_2)   *n13_w[5])
        delta_hw6[2] = X[inx,2]  *  delta_8  *   ((delta_1*delta_2)   *n13_w[5])
        delta_hw6[3] = X[inx,3]  *  delta_8  *   ((delta_1*delta_2)   *n13_w[5])
        delta_hw6[4] = X[inx,4]  *  delta_8  *   ((delta_1*delta_2)   *n13_w[5])
        delta_hw6[5] = X[inx,5]  *  delta_8  *   ((delta_1*delta_2)   *n13_w[5])
        delta_hw6[6] = X[inx,6]  *  delta_8  *   ((delta_1*delta_2)   *n13_w[5])
        delta_hw6[7] = X[inx,7]  *  delta_8  *   ((delta_1*delta_2)   *n13_w[5])
        delta_hw6[8] = X[inx,8]  *  delta_8  *   ((delta_1*delta_2)   *n13_w[5])  

        # neuron n7
        delta_9 = sigmoid(v[6],derive=True)
        # same, need to prop back to weights        
        delta_hw7 = np.ones((9, 1))
        delta_hw7[0] = X[inx,0]  *  delta_9  *   ((delta_1*delta_2)   *n13_w[6])
        delta_hw7[1] = X[inx,1]  *  delta_9  *   ((delta_1*delta_2)   *n13_w[6])
        delta_hw7[2] = X[inx,2]  *  delta_9  *   ((delta_1*delta_2)   *n13_w[6])
        delta_hw7[3] = X[inx,3]  *  delta_9  *   ((delta_1*delta_2)   *n13_w[6])
        delta_hw7[4] = X[inx,4]  *  delta_9  *   ((delta_1*delta_2)   *n13_w[6])
        delta_hw7[5] = X[inx,5]  *  delta_9  *   ((delta_1*delta_2)   *n13_w[6])
        delta_hw7[6] = X[inx,6]  *  delta_9  *   ((delta_1*delta_2)   *n13_w[6])
        delta_hw7[7] = X[inx,7]  *  delta_9  *   ((delta_1*delta_2)   *n13_w[6])
        delta_hw7[8] = X[inx,8]  *  delta_9  *   ((delta_1*delta_2)   *n13_w[6])  

        # neuron n8
        delta_10 = sigmoid(v[7],derive=True)
        # same, need to prop back to weights        
        delta_hw8 = np.ones((9, 1))
        delta_hw8[0] = X[inx,0]  *  delta_10  *   ((delta_1*delta_2)   *n13_w[7])
        delta_hw8[1] = X[inx,1]  *  delta_10  *   ((delta_1*delta_2)   *n13_w[7])
        delta_hw8[2] = X[inx,2]  *  delta_10  *   ((delta_1*delta_2)   *n13_w[7])
        delta_hw8[3] = X[inx,3]  *  delta_10  *   ((delta_1*delta_2)   *n13_w[7])
        delta_hw8[4] = X[inx,4]  *  delta_10  *   ((delta_1*delta_2)   *n13_w[7])
        delta_hw8[5] = X[inx,5]  *  delta_10  *   ((delta_1*delta_2)   *n13_w[7])
        delta_hw8[6] = X[inx,6]  *  delta_10  *   ((delta_1*delta_2)   *n13_w[7])
        delta_hw8[7] = X[inx,7]  *  delta_10  *   ((delta_1*delta_2)   *n13_w[7])
        delta_hw8[8] = X[inx,8]  *  delta_10  *   ((delta_1*delta_2)   *n13_w[7])

        # neuron n9
        delta_11 = sigmoid(v[8],derive=True)
        # same, need to prop back to weights        
        delta_hw9 = np.ones((9, 1))
        delta_hw9[0] = X[inx,0]  *  delta_11  *   ((delta_1*delta_2)   *n13_w[8])
        delta_hw9[1] = X[inx,1]  *  delta_11  *   ((delta_1*delta_2)   *n13_w[8])
        delta_hw9[2] = X[inx,2]  *  delta_11  *   ((delta_1*delta_2)   *n13_w[8])
        delta_hw9[3] = X[inx,3]  *  delta_11  *   ((delta_1*delta_2)   *n13_w[8])
        delta_hw9[4] = X[inx,4]  *  delta_11  *   ((delta_1*delta_2)   *n13_w[8])
        delta_hw9[5] = X[inx,5]  *  delta_11  *   ((delta_1*delta_2)   *n13_w[8])
        delta_hw9[6] = X[inx,6]  *  delta_11  *   ((delta_1*delta_2)   *n13_w[8])
        delta_hw9[7] = X[inx,7]  *  delta_11  *   ((delta_1*delta_2)   *n13_w[8])
        delta_hw9[8] = X[inx,8]  *  delta_11  *   ((delta_1*delta_2)   *n13_w[8])

        # neuron n10
        delta_12 = sigmoid(v[9],derive=True)
        # same, need to prop back to weights        
        delta_hw10 = np.ones((9, 1))
        delta_hw10[0] = X[inx,0]  *  delta_12  *   ((delta_1*delta_2)   *n13_w[9])
        delta_hw10[1] = X[inx,1]  *  delta_12  *   ((delta_1*delta_2)   *n13_w[9])
        delta_hw10[2] = X[inx,2]  *  delta_12  *   ((delta_1*delta_2)   *n13_w[9])
        delta_hw10[3] = X[inx,3]  *  delta_12  *   ((delta_1*delta_2)   *n13_w[9])
        delta_hw10[4] = X[inx,4]  *  delta_12  *   ((delta_1*delta_2)   *n13_w[9])
        delta_hw10[5] = X[inx,5]  *  delta_12  *   ((delta_1*delta_2)   *n13_w[9])
        delta_hw10[6] = X[inx,6]  *  delta_12  *   ((delta_1*delta_2)   *n13_w[9])
        delta_hw10[7] = X[inx,7]  *  delta_12  *   ((delta_1*delta_2)   *n13_w[9])
        delta_hw10[8] = X[inx,8]  *  delta_12  *   ((delta_1*delta_2)   *n13_w[9])

        # neuron n11
        delta_13 = sigmoid(v[10],derive=True)
        # same, need to prop back to weights        
        delta_hw11 = np.ones((9, 1))
        delta_hw11[0] = X[inx,0]  *  delta_13  *   ((delta_1*delta_2)   *n13_w[10])
        delta_hw11[1] = X[inx,1]  *  delta_13  *   ((delta_1*delta_2)   *n13_w[10])
        delta_hw11[2] = X[inx,2]  *  delta_13  *   ((delta_1*delta_2)   *n13_w[10])
        delta_hw11[3] = X[inx,3]  *  delta_13  *   ((delta_1*delta_2)   *n13_w[10])
        delta_hw11[4] = X[inx,4]  *  delta_13  *   ((delta_1*delta_2)   *n13_w[10])
        delta_hw11[5] = X[inx,5]  *  delta_13  *   ((delta_1*delta_2)   *n13_w[10])
        delta_hw11[6] = X[inx,6]  *  delta_13  *   ((delta_1*delta_2)   *n13_w[10])
        delta_hw11[7] = X[inx,7]  *  delta_13  *   ((delta_1*delta_2)   *n13_w[10])
        delta_hw11[8] = X[inx,8]  *  delta_13  *   ((delta_1*delta_2)   *n13_w[10])

        # neuron n12
        delta_14 = sigmoid(v[11],derive=True)
        # same, need to prop back to weights        
        delta_hw12 = np.ones((9, 1))
        delta_hw12[0] = X[inx,0]  *  delta_14  *   ((delta_1*delta_2)   *n13_w[11])
        delta_hw12[1] = X[inx,1]  *  delta_14  *   ((delta_1*delta_2)   *n13_w[11])
        delta_hw12[2] = X[inx,2]  *  delta_14  *   ((delta_1*delta_2)   *n13_w[11])
        delta_hw12[3] = X[inx,3]  *  delta_14  *   ((delta_1*delta_2)   *n13_w[11])
        delta_hw12[4] = X[inx,4]  *  delta_14  *   ((delta_1*delta_2)   *n13_w[11])
        delta_hw12[5] = X[inx,5]  *  delta_14  *   ((delta_1*delta_2)   *n13_w[11])
        delta_hw12[6] = X[inx,6]  *  delta_14  *   ((delta_1*delta_2)   *n13_w[11])
        delta_hw12[7] = X[inx,7]  *  delta_14  *   ((delta_1*delta_2)   *n13_w[11])
        delta_hw12[8] = X[inx,8]  *  delta_14  *   ((delta_1*delta_2)   *n13_w[11])

        # update rule
        n1_w  = n1_w  - eta * delta_hw1   # neuron 1 in hidden layer 1
        n2_w  = n2_w  - eta * delta_hw2   # neuron 2 in hidden layer 1
        n3_w  = n3_w  - eta * delta_hw3   # neuron 3 in hidden layer 1
        n4_w  = n4_w  - eta * delta_hw4   # neuron 4 in hidden layer 1
        n5_w  = n5_w  - eta * delta_hw5   # neuron 5 in hidden layer 1
        n6_w  = n6_w  - eta * delta_hw6   # neuron 6 in hidden layer 1
        n7_w  = n7_w  - eta * delta_hw7   # neuron 7 in hidden layer 1
        n8_w  = n8_w  - eta * delta_hw8   # neuron 8 in hidden layer 1
        n9_w  = n9_w  - eta * delta_hw9   # neuron 9 in hidden layer 1
        n10_w = n10_w - eta * delta_hw10  # neuron 10 in hidden layer 1
        n11_w = n11_w - eta * delta_hw11  # neuron 11 in hidden layer 1
        n12_w = n12_w - eta * delta_hw12  # neuron 12 in hidden layer 1
        n13_w = n13_w - eta * delta_ow    # neuron 13 in output layer
        

print('Ran ' + str(max_epoch) + ' iterations')

fp = open("weights.txt", "w")
fp.write("Weights for neuron 1: \n" + str(n1_w) + "\n\n")
fp.write("Weights for neuron 2: \n" + str(n2_w) + "\n\n")
fp.write("Weights for neuron 3: \n" + str(n3_w) + "\n\n")
fp.write("Weights for neuron 4: \n" + str(n4_w) + "\n\n")
fp.write("Weights for neuron 5: \n" + str(n5_w) + "\n\n")
fp.write("Weights for neuron 6: \n" + str(n6_w) + "\n\n")
fp.write("Weights for neuron 7: \n" + str(n7_w) + "\n\n")
fp.write("Weights for neuron 8: \n" + str(n8_w) + "\n\n")
fp.write("Weights for neuron 9: \n" + str(n9_w) + "\n\n")
fp.write("Weights for neuron 10: \n" + str(n10_w) + "\n\n")
fp.write("Weights for neuron 11: \n" + str(n11_w) + "\n\n")
fp.write("Weights for neuron 12: \n" + str(n12_w) + "\n\n")
fp.write("Weights for neuron 13: \n" + str(n13_w) + "\n\n")
fp.close()


# what were the values (just do forward pass)  
for i in range(6): 
    
    # forward pass
    v = np.ones((12, 1))
    v[0] = np.dot(X[i,:], n1_w)
    v[0] = sigmoid(v[0])
    v[1] = np.dot(X[i,:], n2_w)
    v[1] = sigmoid(v[1])
    v[2] = np.dot(X[i,:], n3_w)
    v[2] = sigmoid(v[2])
    v[3] = np.dot(X[i,:], n4_w)
    v[3] = sigmoid(v[3])
    v[4] = np.dot(X[i,:], n5_w)
    v[4] = sigmoid(v[4])
    v[5] = np.dot(X[i,:], n6_w)
    v[5] = sigmoid(v[5])
    v[6] = np.dot(X[i,:], n7_w)
    v[6] = sigmoid(v[6])
    v[7] = np.dot(X[i,:], n8_w)
    v[7] = sigmoid(v[7]) 
    v[8] = np.dot(X[i,:], n9_w)
    v[8] = sigmoid(v[8]) 
    v[9] = np.dot(X[i,:], n10_w)
    v[9] = sigmoid(v[9]) 
    v[10] = np.dot(X[i,:], n11_w)
    v[10] = sigmoid(v[10]) 
    v[11] = np.dot(X[i,:], n12_w)
    v[11] = sigmoid(v[11]) 
    oo = np.dot(np.transpose(v), n13_w)
    o = sigmoid(oo) 
    print(str(i) + ": produced: " + str(o) + " wanted " + str(y[i]))

# plot it        
# plt.plot(err[0:max_epoch])
# plt.ylabel('error')
# plt.xlabel('epochs')
# plt.show()