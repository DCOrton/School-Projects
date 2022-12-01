from sympy import symbols, pi, sin, cos, simplify
from sympy.matrices import Matrix
import numpy as np
import random 
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
from itertools import product as prod

import torch
import torch.nn.functional as F 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms  
from torch import optim 
from torch import nn  
from torch.utils.data import DataLoader  
from torch.utils.data import TensorDataset  
from tqdm import tqdm  

#DH Paramater Creation
def Dh_param_matrix(a, d, Alpha, Nu):
    # a = Link Length, d = Offset, Alpha = Twist, Nu = Joint Angle
    
    return Matrix( [ [cos(Nu),-sin(Nu)*cos(Alpha), sin(Nu)*sin(Alpha),a*cos(Nu)],
                     [sin(Nu), cos(Nu)*cos(Alpha),-cos(Nu)*sin(Alpha),a*sin(Nu)],
                     [0      , sin(Alpha)        , cos(Alpha)        ,d            ],
                     [0      , 0                 , 0                 ,1            ] ] 
                 ); 
  
def Build_arm_3R(nu1,nu2,nu3):
    #Link Length in meters
    a1 = 0; a2 = 1; a3 = 1;
    #Offset
    d1 = 1; d2 = 0; d3 = 0;
    #Twist
    Alpha1 = pi/2; Alpha2 = 0; Alpha3 = 0
    #Joint Angle
    Nu1 = nu1; Nu2 = nu2; Nu3 = nu3  

    return Dh_param_matrix(a1,d1,Alpha1,Nu1)*Dh_param_matrix(a2,d2,Alpha2,Nu2)*Dh_param_matrix(a3,d3,Alpha3,Nu3);  

  
  
# Generating Dataset  
Nu1, Nu2, Nu3 = symbols('Nu1:4')    
arm = Build_arm_3R(Nu1, Nu2, Nu3)

def f(x):  
    arm_out = arm.subs({Nu1: x[0], Nu2: x[1], Nu3: x[2]})
    arm_out.row_del(3)
    return arm_out,[x[0],x[1],x[2]]

Density = 5
D = 2*Density+1
Random_loss = 0.8
Rlos = int(D*Random_loss)

n = np.random.choice( np.linspace(-np.pi,np.pi,D), Rlos , replace=False)
result, sequence = zip(*[f(E) for E in list(prod(n, repeat=3)) ])

DF = pd.DataFrame.from_records(result)


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size = 12
hidden_size = 256
num_layers = 2
num_classes = 3
sequence_length = 512
learning_rate = 0.005
batch_size = 64
num_epochs = 100


class RNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Set initial hidden and cell states        
        h0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm( x, (h0, c0) ) 
        out = out.reshape(out.shape[0], -1)
        
        out = self.fc(out)
        return out


# Load Data
x_train = np.array(DF, dtype=np.float32)
y_train = np.array(sequence, dtype=np.float32)

inputs = torch.asarray(x_train, dtype=torch.float32)
targets = torch.asarray(y_train, dtype=torch.float32)

train_dataset = TensorDataset(inputs, targets)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Initialize network
model = RNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)
                
        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent update step/adam step
        optimizer.step()
        
    if (epoch+1) % 10 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

        
#Colour Gradient stuff        
def hex_to_RGB(hex_str):
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_colour_gradient(c1, c2, n):
    assert n > 1
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]        

colour1 = "#A0E4CB"
colour2 = "#59C1BD"
colour3 = "#FFE15D"
colour4 = "#F49D1A"

grad1 = get_colour_gradient(colour1, colour2, len(x1))
grad2 = get_colour_gradient(colour3, colour4, len(x1))


#Checking Model        
predicted = pd.DataFrame(model(torch.from_numpy(x_train)).detach().numpy() )
verif = pd.DataFrame(y_train)

x1 = np.linspace(1, 512, num=512);

#Creating four polar graphs but only need 3
fig, axs = plt.subplots(2, 2, subplot_kw=dict(projection="polar") , figsize=(20, 15))
axs[0, 0].scatter(x1, verif.iloc[:,0].to_numpy(), c=grad1, marker='o', cmap='hsv', alpha=0.95)
axs[0, 1].scatter(x1, verif.iloc[:,1].to_numpy(), c=grad1, marker='o', cmap='hsv', alpha=0.95)
axs[1, 0].scatter(x1, verif.iloc[:,2].to_numpy(), c=grad1, marker='o', cmap='hsv', alpha=0.95)

axs[0, 0].scatter(x1, predicted.iloc[:,0].to_numpy(), c=grad2, marker='x', cmap='hsv', alpha=0.95)
axs[0, 1].scatter(x1, predicted.iloc[:,1].to_numpy(), c=grad2, marker='x', cmap='hsv', alpha=0.95)
axs[1, 0].scatter(x1, predicted.iloc[:,2].to_numpy(), c=grad2, marker='x', cmap='hsv', alpha=0.95)

plt.show()
