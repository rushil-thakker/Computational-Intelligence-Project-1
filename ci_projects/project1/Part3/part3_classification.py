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
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from sklearn.utils import shuffle
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
            nn.Sigmoid(),                     ###CHANGE ACTIVATION FUNCTIONS
            nn.Linear(H, D_out),
            nn.Sigmoid()
        )        
    def forward(self, x):
        y_pred = self.layers(x)
        return y_pred
############################################

D_in, H, D_out = 9, 9, 1    
net = AnotherMLP(D_in, H, D_out)

def criterion(out,label):
    return (label - out)**2

############################################
# threw in a new learner for you!
############################################
optimizer = optim.Adam(net.parameters(), lr=1e-1)       ###CHOOSING NEW LEARNERS

############################################
# Begin reading in data
############################################

fp = open('breast-cancer.data', 'r')
inputs = np.zeros((277, 9))                  #input Data
L = np.zeros((277, 1))                  #labels
row = 0

for line in fp.readlines():
    data = line.strip()
    data = line.split(',')

    classification = data[0]

    if(classification == "no-recurrence-events"):
        classification = 0
    elif(classification == "recurrence-events"):
        classification = 1

    age = data[1]

    if(age == "10-19"):
        age = 0
    elif(age == "20-29"):
        age = 1
    elif(age == "30-39"):
        age = 2
    elif(age == "40-49"):
        age = 3
    elif(age == "50-59"):
        age = 4
    elif(age == "60-69"):
        age = 5
    elif(age == "70-79"):
        age = 6
    elif(age == "80-89"):
        age = 7
    elif(age == "90-99"):
        age = 8

    menopause = data[2]

    if(menopause == "lt40"):
        menopause = 0
    elif(menopause == "ge40"):
        menopause = 1
    elif(menopause == "premeno"):
        menopause = 2

    tumor_size = data[3]

    if(tumor_size == "0-4"):
        tumor_size = 0
    elif(tumor_size == "5-9"):
        tumor_size = 1
    elif(tumor_size == "10-14"):
        tumor_size = 2
    elif(tumor_size == "15-19"):
        tumor_size = 3
    elif(tumor_size == "20-24"):
        tumor_size = 4
    elif(tumor_size == "25-29"):
        tumor_size = 5
    elif(tumor_size == "30-34"):
        tumor_size = 6
    elif(tumor_size == "35-39"):
        tumor_size = 7
    elif(tumor_size == "40-44"):
        tumor_size = 8
    elif(tumor_size == "45-49"):
        tumor_size = 9
    elif(tumor_size == "50-54"):
        tumor_size = 10
    elif(tumor_size == "55-59"):
        tumor_size = 11

    inv_nodes = data[4]

    if(inv_nodes == "0-2"):
        inv_nodes = 0
    elif(inv_nodes == "3-5"):
        inv_nodes = 1
    elif(inv_nodes == "6-8"):
        inv_nodes = 2
    elif(inv_nodes == "9-11"):
        inv_nodes = 3
    elif(inv_nodes == "12-14"):
        inv_nodes = 4
    elif(inv_nodes == "15-17"):
        inv_nodes = 5
    elif(inv_nodes == "18-20"):
        inv_nodes = 6
    elif(inv_nodes == "21-23"):
        inv_nodes = 7
    elif(inv_nodes == "24-26"):
        inv_nodes = 8
    elif(inv_nodes == "27-29"):
        inv_nodes = 9
    elif(inv_nodes == "30-32"):
        inv_nodes = 10
    elif(inv_nodes == "33-35"):
        inv_nodes = 11
    elif(inv_nodes == "36-39"):
        inv_nodes = 12

    node_caps = data[5]

    if(node_caps == "yes"):
        node_caps = 1
    elif(node_caps == "no"):
        node_caps = 0

    deg_malig = data[6]
    deg_malig = int(deg_malig)

    breast = data[7]

    if(breast == "left"):
        breast = 0
    elif(breast == "right"):
        breast = 1

    breast_quad = data[8]

    if(breast_quad == "left_up"):
        breast_quad = 0
    elif(breast_quad == "left_low"):
        breast_quad = 1
    elif(breast_quad == "right_up"):
        breast_quad = 2
    elif(breast_quad == "right_low"):
        breast_quad = 3
    elif(breast_quad == "central"):
        breast_quad = 4

    irradiat = data[9]
    if(irradiat == "yes\n" or irradiat == "yes"):
        irradiat = 1
    elif(irradiat == "no\n" or irradiat == "no"):
        irradiat = 0

    L[row][0] = classification 
    inputs[row][0] = age
    inputs[row][1] = menopause
    inputs[row][2] = tumor_size
    inputs[row][3] = inv_nodes
    inputs[row][4] = node_caps
    inputs[row][5] = deg_malig
    inputs[row][6] = breast
    inputs[row][7] = breast_quad
    inputs[row][8] = irradiat

    row += 1

fp.close()

L = torch.from_numpy(np.array(L))               #numpy to tensor
inputs = torch.from_numpy(np.array(inputs))     #numpy to tensor

############################################
# End reading in data
############################################
np.random.seed(333)                         #seed for shuffling
inputs, L = shuffle(inputs, L)              #shuffle data 
train = 247
test = 30

net = net.float()                           #allow net to handle floating pt inputs

for epoch in tqdm(range(100)):
    for i in range(train):                            #90%
        X = Variable(inputs[i,:])                 #aux for forward pass
        Y = Variable(L[i])                      #aux for forward pass
        optimizer.zero_grad()
        outputs = net(X.float())                        #forward pass
        loss = criterion(outputs, Y)                    #forward pass
        loss.backward()
        optimizer.step()

values = np.zeros(test,)                  #numpy array to hold prediction values
actual = np.zeros(test,)                  #numpy array to hold actual labels

for i in range(test):                             #10%
    X = Variable(inputs[train+i,:])   
    Y = Variable(L[train+i])
    outputs = net(X.float())
    outputs = outputs.detach().numpy()          #change from tensor to numpy
    loss += criterion(outputs, Y)
    print(str(i) + ": produced: " + str(outputs) + " wanted " + str(L[train+i]))
    values[i] = outputs.round()                 #confusion matrix needs discrete values
    actual[i] = Y

print("The loss is: " + str(loss/test) + "\n")


#plotting confusion matrix using SciKit
conf_matrix = confusion_matrix(y_true=actual, y_pred=values)
fig, ax = plt.subplots(figsize=(7.5, 7.5))
ax.matshow(conf_matrix, cmap=plt.cm.winter, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()