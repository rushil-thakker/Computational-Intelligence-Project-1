import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Function
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
np.set_printoptions(threshold=sys.maxsize)

############################################
# Here is what's different in this code!!!
#   self.layers = nn.Sequential
############################################
class AnotherMLP(nn.Module):
    def __init__(self, D_in, H, D_out):
        super(AnotherMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(D_in, H),
            nn.Sigmoid(),  # nn.ReLU()                     ###CHANGE ACTIVATION FUNCTIONS
            nn.Linear(H, D_out),
            nn.ReLU()
        )        
    def forward(self, x):
        y_pred = self.layers(x)
        return y_pred
############################################

D_in, H, D_out = 8, 8, 1    
net = AnotherMLP(D_in, H, D_out)

def criterion(out,label):
    return (label - out)**2

############################################
# threw in a new learner for you!
############################################
optimizer = optim.SGD(net.parameters(), lr=1e-1)       ###CHOOSING NEW LEARNERS

############################################
# Begin reading in data
############################################

fp = open('toxicity.csv', 'r')
inputs = np.zeros((546, 8))                  #input Data
L = np.zeros((546, 1))                  #labels
row = 0

for line in fp.readlines():
    data = line.strip()
    data = line.split(';')

    TPSA = data[0]
    TPSA = float(TPSA)

    SAacc = data[1]
    SAacc = float(SAacc)

    H050 = data[2]
    H050 = float(H050)

    MLOGP = data[3]
    MLOGP = float(MLOGP)

    RDCHI = data[4]
    RDCHI = float(RDCHI)

    GATS1p = data[5]
    GATS1p = float(GATS1p)

    nN = data[6]
    nN = float(nN)

    C040 = data[7]
    C040 = float(C040)

    LC50 = data[8]
    LC50 = float(LC50)

    L[row][0] = LC50 
    inputs[row][0] = TPSA
    inputs[row][1] = SAacc
    inputs[row][2] = H050
    inputs[row][3] = MLOGP
    inputs[row][4] = RDCHI
    inputs[row][5] = GATS1p
    inputs[row][6] = nN
    inputs[row][7] = C040
    row += 1

fp.close()

L = torch.from_numpy(np.array(L))               #numpy to tensor
inputs = torch.from_numpy(np.array(inputs))     #numpy to tensor

############################################
# End reading in data
############################################
net = net.float()                           #handle float inputs
train = 500
test = 46
loss_sum = 0
epoch = 10
train_error = np.zeros(epoch,)
test_error = np.zeros(test,)

for epoch in tqdm(range(epoch)):
    for i in range(train):                            #90%
        X = Variable(inputs[i,:])                    #aux for forward pass
        Y = Variable(L[i])                          #aux for forward pass
        optimizer.zero_grad()
        outputs = net(X.float())                        #forward pass
        loss = criterion(outputs, Y)                    #forward pass
        loss_sum += loss
        loss.backward()
        optimizer.step()

    train_error[epoch] = loss_sum/(train*epoch)


for i in range(test):                             #10%
    X = Variable(inputs[train+i,:])   
    Y = Variable(L[train+i])
    outputs = net(X.float())                    #forward pass
    outputs = outputs.detach().numpy()          #change back into numpy from tensor
    temp = criterion(outputs, Y)
    loss += temp
    test_error[i] = temp
    print(str(i) + ": produced: " + str(outputs) + " wanted " + str(L[train+i]))

print("The loss is: " + str(loss/test) + "\n")

#plot error vs epochs
plt.plot(train_error)
plt.ylabel('Error')
plt.xlabel('Epoch')
plt.title('Convergence Plot')
plt.show()