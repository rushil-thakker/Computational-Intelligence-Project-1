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

# weights with random numbers for green neurons
np.random.seed(666)
green_weights = np.random.rand(6,9)

# weights with random numbers for orange neurons
np.random.seed(333)
orange_weights = np.random.rand(6,9)

#combine into a 12x9 weight matrix
base_weights = np.concatenate((green_weights, orange_weights), axis=0)

# weights with random numbers for ouput neuron, 12x1 matrix
np.random.seed(111)
n13_w = np.random.rand(12)

print('hidden layer 1, neuron 1 weights')
print(base_weights[0])
print('hidden layer 1, neuron 2 weights')
print(base_weights[1])
print('hidden layer 1, neuron 3 weights')
print(base_weights[2])
print('hidden layer 1, neuron 4 weights')
print(base_weights[3])
print('hidden layer 1, neuron 5 weights')
print(base_weights[4])
print('hidden layer 1, neuron 6 weights')
print(base_weights[5])
print('hidden layer 2, neuron 1 weights')
print(base_weights[6])
print('hidden layer 2, neuron 2 weights')
print(base_weights[7])
print('hidden layer 2, neuron 3 weights')
print(base_weights[8])
print('hidden layer 2, neuron 4 weights')
print(base_weights[9])
print('hidden layer 2, neuron 5 weights')
print(base_weights[10])
print('hidden layer 2, neuron 6 weights')
print(base_weights[11])
print('output layer, neuron 1 weights')
print(n13_w)

###############################################
# Epochs
###############################################
eta = 2 # learning rate
max_epoch = 1000 # how many epochs? (each epoch will run through all 4 data points)
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
        v = np.matmul(base_weights, X[inx,:].transpose())
        oo = np.matmul(v.transpose(), n13_w)
        o = sigmoid(oo)
        
        # error
        err[k] = err[k] + (np.power((y[inx] - o), 2.0))

        # backprop time folks!!!
        
        # output layer, our delta is (delta_1 * delta_2)
        delta_1 = (-1.0) * (y[inx] - o)
        delta_2 = sigmoid(o,derive=True) # note how I called it, I passed o=sigmoid(oo)

        #(Err derivative * Sigmoid derivative)
        delta = (delta_1*delta_2)
        
        # now, lets prop it back to the weights
        delta_ow = np.ones(12)

        # format is
        #  delta_index = (input to final neuron) * (Err derivative * Sigmoid derivative)
        delta_ow = np.multiply(v, delta)

        #holds the derivative of the sigmoid for all v
        sigmoid_der = sigmoid(v, derive=True)

        # = weight to output neuron * error from output
        weights_error = np.multiply(n13_w, delta)

        # = sig der * weight to ouput neuron * error from output
        neuron_error = np.matmul(sigmoid_der, weights_error)

        # update rule
        base_weights[0]  = base_weights[0]  - eta * np.multiply(neuron_error, X[inx,:])   # neuron 1 in hidden layer 1
        base_weights[1]  = base_weights[1]  - eta * np.multiply(neuron_error, X[inx,:])   # neuron 2 in hidden layer 1
        base_weights[2]  = base_weights[2]  - eta * np.multiply(neuron_error, X[inx,:])   # neuron 3 in hidden layer 1
        base_weights[3]  = base_weights[3]  - eta * np.multiply(neuron_error, X[inx,:])   # neuron 4 in hidden layer 1
        base_weights[4]  = base_weights[4]  - eta * np.multiply(neuron_error, X[inx,:])   # neuron 5 in hidden layer 1
        base_weights[5]  = base_weights[5]  - eta * np.multiply(neuron_error, X[inx,:])   # neuron 6 in hidden layer 1
        base_weights[6]  = base_weights[6]  - eta * np.multiply(neuron_error, X[inx,:])   # neuron 7 in hidden layer 1
        base_weights[7]  = base_weights[7]  - eta * np.multiply(neuron_error, X[inx,:])   # neuron 8 in hidden layer 1
        base_weights[8]  = base_weights[8]  - eta * np.multiply(neuron_error, X[inx,:])   # neuron 9 in hidden layer 1
        base_weights[9]  = base_weights[9]  - eta * np.multiply(neuron_error, X[inx,:])   # neuron 10 in hidden layer 1
        base_weights[10] = base_weights[10] - eta * np.multiply(neuron_error, X[inx,:])   # neuron 11 in hidden layer 1
        base_weights[11] = base_weights[11] - eta * np.multiply(neuron_error, X[inx,:])   # neuron 12 in hidden layer 1
        n13_w            = n13_w            - eta * delta_ow                              # neuron 13 in output layer
        

print('Ran ' + str(max_epoch) + ' iterations')
        
# what were the values (just do forward pass)  
for i in range(6): 
    
    # forward pass
    v = np.ones((12, 1))
    v = np.matmul(base_weights, X[inx,:].transpose())
    oo = np.matmul(v.transpose(), n13_w)
    o = sigmoid(oo)
    print(str(i) + ": produced: " + str(o) + " wanted " + str(y[i]))

# plot it        
# plt.plot(err[0:max_epoch])
# plt.ylabel('error')
# plt.xlabel('epochs')
# plt.show()