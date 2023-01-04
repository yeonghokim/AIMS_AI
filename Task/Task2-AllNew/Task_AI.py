# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 15:51:07 2023

@author: yeong
"""
#%% Parameter
CNNinput_width=30

#%% csv data read
import numpy as np
import csv
'''
Traindata = np.empty((0,2,CNNinput_width), float)

Trainlabel = np.empty((0,4), float)

Testdata = np.empty((0,2,CNNinput_width), float)
Testlabel = np.empty((0,4), float)
path ="C:/Users/yeong/Desktop/ResearchStudent/Github/AIMS_AI/Task/Task2-AllNew/"
f = open(path+"Traindata.csv", "r")
rdr = csv.reader(f)
for line in rdr:
    HS=np.array(line[:CNNinput_width*2])
    Traindata=np.insert(Traindata,len(Traindata),np.reshape(HS,(2,CNNinput_width)),axis=0)
f.close()

f = open(path+"Trainlabel.csv", "r")
rdr = csv.reader(f)
for line in rdr:
    Trainlabel=np.insert(Trainlabel,len(Trainlabel),np.array(line[:4]),axis=0)
f.close()

f = open(path+"Testdata.csv", "r")
rdr = csv.reader(f)
for line in rdr:
    HS=np.array(line[:CNNinput_width*2])
    Testdata=np.insert(Testdata,len(Testdata),np.reshape(HS,(2,CNNinput_width)),axis=0)

f.close()

f = open(path+"Testlabel.csv", "r")
rdr = csv.reader(f)
for line in rdr:
    Testlabel=np.insert(Testlabel,len(Testlabel),np.array(line[:4]),axis=0)

f.close()
'''
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import platform

device=''
if platform.system()=='Windows':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Windows cuda')
elif platform.system()=='Linux':
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    print('Linux cuda')
else:
    exit()

class Custom_dataset(Dataset):
    def __init__(self, Xdata, label, transform=None):
        self.len = len(Xdata)
        self.Xdata = torch.FloatTensor(Xdata).to(device)
        self.labels = torch.FloatTensor(label).to(device)
        self.transform = transform
    
    def __len__(self):
        return self.len    
    
    def __getitem__(self,idx):
        return self.Xdata[idx],self.labels[idx] 


BATCH_SIZE=32

train_dataset = Custom_dataset(Traindata, Trainlabel)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # # 배치 형태로 만들어 주자.

test_dataset = Custom_dataset(Testdata, Testlabel)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True) # # 배치 형태로 만들어 주자.
#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(2, 4, 5,padding=3) #
        self.pool1 = nn.MaxPool1d(2,1) #
        self.bn1 = nn.BatchNorm1d(4) #
        self.conv2 = nn.Conv1d(4, 8, 5,padding=3) #
        self.bn2 = nn.BatchNorm1d(8) #
        self.fc1 = nn.Linear(8 * (CNNinput_width+2), 350) # 
        self.fc2 = nn.Linear(350, 80) # 
        self.fc3 = nn.Linear(80, 4)
   
    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x)))) # conv1 -> ReLU -> pool1
        x = self.pool1(F.leaky_relu(self.bn2(self.conv2(x)))) # conv2 -> ReLU -> pool2
        
        x = x.view(-1, 8 * (CNNinput_width+2)) # 5x5 피쳐맵 16개를 일렬로 만든다.
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        
        return x
    
net = Net().to(device)

PATH = './CNN_Net3.pth'
net.load_state_dict(torch.load(PATH, map_location=torch.device(device)))

import torch.optim as optim

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(),lr=0.001)

for epoch in range(170):
    running_loss = 0.0
    for i, data in enumerate(train_loader,0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()

        outputs = net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss +=loss.item()
    print('[%d] loss: %.3f' %(epoch + 1, running_loss / BATCH_SIZE))    

PATH = './CNN_Net3.pth'
torch.save(net.state_dict(),PATH)        
#%%
def NRMSE(p,y):
    RMSE = torch.sqrt((torch.mean((p-y)**2)))/(torch.max(y) - torch.min(y))
    return RMSE

import PhaseCheck as Pc
import math

correct = 0
total = 0
desired_accuracy=0.2
test_labellist =list();
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
        
        L = outputs[0][0]*outputs[0][0]+outputs[0][1]*outputs[0][1]
        R = outputs[0][2]*outputs[0][2]+outputs[0][3]*outputs[0][3]
        
        L=math.sqrt(L)
        R=math.sqrt(R)
        
        outputs[0][0]=outputs[0][0]/L
        outputs[0][1]=outputs[0][1]/L
        outputs[0][2]=outputs[0][2]/R
        outputs[0][3]=outputs[0][3]/R
        
        total+=labels.size(0)
        if  NRMSE(outputs,labels)<desired_accuracy:
            correct += 1
        
print('Accuracy of the network on the %d test : %d %%'%(total,100*correct/total))
