# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 20:14:03 2022

@author: yeong
"""
import math
import numpy as np

filepath = "C:/Users/yeong/Desktop/학부연구생/Task 모음/Task2/data/sub1"
ex_type= ["W","R"]
ex_speed = ["fast","mid","slow"]
ex_feet = ["L","R"]
ex_data = ["angle","HS"]

Traindata_L_Angle=list() #input
Traindata_L_HS =list() #output
Traindata_L_Changed_HS =list() #output

Traindata_R_Angle=list() #input
Traindata_R_HS =list() #output
Traindata_R_Changed_HS =list() #output
# E1-Wf

exnum = 1
extype = 0 # 0 = "W"  1 ="R"
exspeed = 0 # 0 = "fast"  1 ="slow"  2="mid"
exfeet = 0 # 0="L" 1="R"
exdata=0  # 0="angle" 1="HS"

#파일 불러오는 부분
for exnum in range(1,3):
    for extype in range(0,2):
        i=2
        if extype==0:
            i=i+1
        for exspeed in range(0,i):
            for exfeet in range(0,2):
                for exdata in range(0,2):
                    fdata=0
                    with open(filepath+"_T"+str(exnum)+"_"+ex_type[extype]+"_"+ex_speed[exspeed]+"_"+ex_feet[exfeet]+"_"+ex_data[exdata]+".txt", "r") as f:
                        fdata = f.readlines()
                        if exdata==0:
                            fdata = list(map(float,fdata))
                        else:
                            fdata = list(map(int,fdata))
                    
                    if exfeet==0:
                        if exdata==0:
                            Traindata_L_Angle.append(fdata)
                        else:
                            Traindata_L_HS.append(fdata)
                    else:
                        if exdata==0:
                            Traindata_R_Angle.append(fdata)
                        else:
                            Traindata_R_HS.append(fdata)        

#각도로 된 리스트를 좌표 튜플들의 리스트로 바꿔주는 함수

def changeCoor(li):
    exli = list()
    for i in li:
        if i==0 or i==360:
             exli.append((1,0))       
        elif i==90:
            exli.append((0,1))
        elif i==180:
             exli.append((-1,0))       
        elif i==270:
            exli.append((0,-1))
        else:
            exli.append((math.cos(math.radians(i)),math.sin(math.radians(i))))
        
    return exli
        
#10개의 input을 각도로 변경
for i in range(0,10):
    listL = Traindata_L_HS[i]
    listR = Traindata_R_HS[i]
    firstL=listL[0]
    firstR=listR[0]
    
    ChangedlistL=list()
    ChangedlistR=list()
    ChangedlistL.append(0)
    ChangedlistR.append(0)
    
    #Left foot Theta
    for listflag in range(1,len(listL)):
        endL = listL[listflag]
        width = endL-firstL
        
        for j in range(1,width+1):
            ChangedlistL.append(j*360/width)
        
        firstL=endL
      
    #Right foot Theta  
    for listflag in range(1,len(listR)):
        endR = listR[listflag]
        width = endR-firstR
        
        for j in range(1,width+1):
            ChangedlistR.append(j*360/width)
        
        firstR=endR  
    
    firstR=listR[0]
    ChangedlistL=ChangedlistL[firstR:]
    
    Traindata_L_Changed_HS.append(changeCoor(ChangedlistL))
    Traindata_R_Changed_HS.append(changeCoor(ChangedlistR))


CNNinput_width=100

#input데이터를 크기(CNNinput_width)를 이용해 변환

Traindatalist=list()
Trainlabellist=list()

Testdatalist=list()
Testlabellist=list()

for i in range(0,10):
    HSLeft = np.array(Traindata_L_Changed_HS[i])
    HSRight = np.array(Traindata_R_Changed_HS[i])
    AngleLeft = np.array(Traindata_L_Angle[i])
    AngleRight = np.array(Traindata_R_Angle[i])
    
    minlen = min(len(HSLeft),len(HSRight))
    for flag in range(0,minlen-CNNinput_width+1):
        anglelist = np.concatenate((AngleLeft[flag:flag+CNNinput_width],AngleRight[flag:flag+CNNinput_width]))
        anglelist = anglelist.reshape(2,-1)
        hslist =  np.concatenate((HSLeft[flag+CNNinput_width-1],HSRight[flag+CNNinput_width-1]))
        if i<7:
            Traindatalist.append(anglelist.tolist())
            Trainlabellist.append(hslist.tolist())
        else:
            Testdatalist.append(anglelist.tolist())
            Testlabellist.append(hslist.tolist())


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class MyIMUData(Dataset):
    def __init__(self,imudata,label):
        #리스트로 다 들어옴
        self.imudata = imudata # 2 X 10
        self.label = label # 4
        self.len = self.label.shape[0]
        
    def __getitem__(self,index):
        return self.imudata[index], self.label[index]
    
    def __len__(self):
        return self.len

BATCH_SIZE=32
train_data = MyIMUData(torch.tensor(Traindatalist).to(device), torch.tensor(Trainlabellist).to(device)) # train 데이터를 불러와주고,
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True) # # 배치 형태로 만들어 주자.

test_data = MyIMUData(torch.tensor(Testdatalist).to(device), torch.tensor(Testlabellist).to(device)) # train 데이터를 불러와주고,
test_loader = DataLoader(test_data, batch_size=1, shuffle=True) # # 배치 형태로 만들어 주자.



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(2, 4, 5,padding=2) #
        self.pool1 = nn.MaxPool1d(3,1) #
        self.bn1= nn.BatchNorm1d(4)
        self.conv2 = nn.Conv1d(4, 8, 5,padding=2) #
        self.bn2= nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(8 * (CNNinput_width-2), 350) # 
        self.fc2 = nn.Linear(350, 80) # 
        self.fc3 = nn.Linear(80, 4)
   
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x))) # conv1 -> ReLU -> pool1
        x = self.pool1(F.leaky_relu(self.bn2(self.conv2(x)))) # conv2 -> ReLU -> pool2

        x = x.view(-1, 8 * (CNNinput_width-2)) # 5x5 피쳐맵 16개를 일렬로 만든다.
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        
        
        return x

net = Net()
net = net.to(device)

#PATH = './CNN_Net.pth'
#net.load_state_dict(torch.load(PATH))

import torch.optim as optim

criterion = nn.MSELoss().to(device)
optimizer = optim.Adam(net.parameters(),lr=0.003)

for epoch in range(200):
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
     

PATH = './CNN_Net.pth'
torch.save(net.state_dict(),PATH)

def NRMSELoss(data1, data2):
    criterion = nn.MSELoss()
    loss = torch.sqrt(criterion(data1, data2))/(torch.max(data2)-torch.min(data1))
    return loss
    
    
correct = 0
total = 0
desired_accuracy=0.03
test_labellist =list();
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = net(inputs)
            
        total+=labels.size(0)
        for i in range(0,outputs.size()[0]):
            test_labellist.append(criterion(outputs,labels).tolist())
            if  NRMSELoss(outputs,labels)<desired_accuracy:
                correct += 1
        
print('Accuracy of the network on the %d test : %d %%'%(total,100*correct/total))

PATH = './CNN_Net3.pth'
torch.save(net.state_dict(),PATH)
