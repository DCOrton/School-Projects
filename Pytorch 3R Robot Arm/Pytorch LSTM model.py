// DCOrton
// Nov. 5th, 2022


from sympy import symbols, pi, sin, cos, simplify
from sympy.matrices import Matrix
import numpy as np
import random 
import matplotlib.pyplot as plt
import pandas as pd
import time
import math

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


from itertools import product as prod
Nu1, Nu2, Nu3 = symbols('Nu1:4')    
arm = Build_arm_3R(Nu1, Nu2, Nu3)

def f(x):    
    return arm.subs({Nu1: x[0], Nu2: x[1], Nu3: x[2]}),[x[0],x[1],x[2]]

Density = 10
D = 2*Density+1
Random_loss = 0.8
Rlos = int(D*Random_loss)

n = np.random.choice( np.linspace(-np.pi,np.pi,D), Rlos , replace=False)
result, sequence = zip(*[f(E) for E in list(prod(n, repeat=3)) ])
            
DF = pd.DataFrame.from_records(result)


import ipywidgets as widgets

elev = widgets.FloatSlider(description='elev',min=0,max=90,step=15)
azim = widgets.FloatSlider(description='azim',min=0,max=90,step=15)

def f(elevation, azimuth):    
    fig2 = plt.figure(figsize=(20, 15))
    ax2 = plt.axes(projection='3d', elev=elevation, azim=azimuth )
    ax2.scatter3D(DF.iloc[:,0], DF.iloc[:,1], DF.iloc[:,2], c='b');
    ax2.scatter3D(DF.iloc[:,3], DF.iloc[:,4], DF.iloc[:,5], c='r');
    
out = widgets.interactive_output(f, {'elevation': elev,'azimuth': azim})

widgets.VBox( [ out, elev, azim ] )


import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyper-parameters
num_epochs = 100
learning_rate = 0.001

x_train = np.array(DF, dtype=np.float32)
y_train = np.array(sequence, dtype=np.float32)

L, h = x_train.shape
print(L)
print(h)

print(y_train.shape)

model = LSTMtoLinear( input_size=h, hidden_size=h, num_layers=2, linear_output=3)
    
#nn.LSTM(input_size=h, hidden_size=h, num_layers=3),
#nn.Linear(h, 3)


# Loss and optimizer
criterion = nn.MSELoss() #MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  #SGD

inputs = torch.asarray(x_train, dtype=torch.float32)
targets = torch.asarray(y_train, dtype=torch.float32)

# Train the model
for epoch in range(num_epochs):

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Save the model checkpoint
#torch.save(model.state_dict(), 'model.ckpt')