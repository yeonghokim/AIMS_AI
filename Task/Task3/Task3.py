# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 14:20:56 2023

@author: yeong
"""

import numpy as np
import pandas as pd
import math
import torch
from torch.utils.data import Dataset, DataLoader

#%% platform cuda & path setting
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
    
Path=''
Linux_filepath = "/data/yeongho/task3/data/sub_3/"
Window_filepath = "C:/Users/yeong/Desktop/ResearchStudent/Github/AIMS_AI/data/Task3/GaitData/sub_3/"
filepath=''
if platform.system()=='Windows':
    Path = Window_filepath
    print('Windows filepath')
elif platform.system()=='Linux':
    Path = Linux_filepath
    print('Linux filepath')
else:
    exit()

#%% HyperParameter
WindowSize = 30
NormalTrial = 20


#%% 수술 수정 20220805
def Gait_label(HS_L , HS_R): 
            
    if HS_R[0] < HS_L[0] :
        
        stridetimeL = np.diff(HS_L)
        stridetimeR = np.diff(HS_R)
        
        k = int(min(HS_L[-1],HS_R[-1])-HS_L[0]+1)
        
        true_phase = np.zeros((4,k))
        
        
        phasedataL = np.zeros((1))
        for i in range(len(stridetimeL)):
            pdata = np.linspace(0,1,int(stridetimeL[i]+1))
            pdata = np.delete(pdata, [-1])
            phasedataL = np.append(phasedataL,pdata)
            
        phasedataL = np.delete(phasedataL, [0])    
        phasedataL = np.append(phasedataL,1)   
        
        phasedataL = phasedataL[:k]
        theta = phasedataL*2*math.pi
        
        x = np.cos(theta)
        y = np.sin(theta)
        
        true_phase[0,:] = x
        true_phase[1,:] = y 
        
        phasedataR = np.zeros((1))
        
        for i in range(len(stridetimeR)):
            pdata = np.linspace(0,1,int(stridetimeR[i]+1))
            pdata = np.delete(pdata, [-1])
            phasedataR = np.append(phasedataR,pdata)
            
        phasedataR = np.delete(phasedataR, [0])
        phasedataR = np.append(phasedataR ,1)

        phasedataR = phasedataR[int(HS_L[0])-int(HS_R[0]):k+int(HS_L[0])-int(HS_R[0])]
        theta = phasedataR*2*math.pi
        
        x = np.cos(theta)
        y = np.sin(theta)
       
        true_phase[2,:] = x
        true_phase[3,:] = y    
              
    else :           
        
        stridetimeL = np.diff(HS_L)
        stridetimeR = np.diff(HS_R)
        
        k = int(min(HS_L[-1],HS_R[-1])-HS_R[0]+1)
        
        true_phase = np.zeros((4,k))
        
        
        phasedataL = np.zeros((1))
     
        for i in range(len(stridetimeL)):
            pdata = np.linspace(0,1,int(stridetimeL[i]+1))
            pdata = np.delete(pdata, [-1])
            phasedataL = np.append(phasedataL,pdata)
            
        phasedataL = np.delete(phasedataL, [0])
        phasedataL = np.append(phasedataL ,1)

        phasedataL = phasedataL[int(HS_R[0])-int(HS_L[0]):k+int(HS_R[0])-int(HS_L[0])]
        theta = phasedataL*2*math.pi
        
        x = np.cos(theta)
        y = np.sin(theta)
          
        true_phase[0,:] = x
        true_phase[1,:] = y    
        
        phasedataR = np.zeros((1))

        for i in range(len(stridetimeR)):
            pdata = np.linspace(0,1,int(stridetimeR[i]+1))
            pdata = np.delete(pdata, [-1])
            phasedataR = np.append(phasedataR,pdata)
            
        phasedataR = np.delete(phasedataR, [0])    
        phasedataR = np.append(phasedataR,1)   
        
        phasedataR = phasedataR[:k]
        theta = phasedataR*2*math.pi
        
        x = np.cos(theta)
        y = np.sin(theta)
            
        true_phase[2,:] = x
        true_phase[3,:] = y 
        
    return true_phase

#%%넘파이 수정
def window(windowsize,shiftsize,X,Y):
    X_out = np.zeros((windowsize,X.shape[1],X.shape[0]-windowsize+1)) # 윈도우사이즈 값개수 데이터개수
    Y_out = np.zeros(Y.shape)
    #Y_out = np.zeros((Y.shape[0],Y.shape[1])) 
    
    for i in range(0,X.shape[0]-windowsize+1,shiftsize):
        for j in range(X.shape[1]):
            X_out[:,j,i] = X[i:i+windowsize,j]
    for i in range(0,Y.shape[1],shiftsize):
        Y_out[:,i] = Y[:,i]
    
    return X_out, Y_out

#%% Read csv normal data

x_test_n = np.zeros((30,12,1))
x_train_n = np.zeros((30,12,1))
y_test_n = np.zeros((4,1))
y_train_n = np.zeros((4,1))



for trial in range(NormalTrial):
    datas = pd.read_csv(Path+'sub_3_trial'+str(trial+1)+'.csv')
    hsL = pd.read_csv(Path+'hs_l_sub_3_trial'+str(trial+1)+'.csv')
    hsR = pd.read_csv(Path+'hs_r_sub_3_trial'+str(trial+1)+'.csv')
    
    datas = datas.to_numpy()
    
    hsL = hsL.to_numpy().reshape(-1)
    hsR = hsR.to_numpy().reshape(-1)
    
    stridetimeL = np.diff(hsL)
    
    #hs기준으로 정리
    start1 = min(hsL[0],hsR[0])
    start2 = max(hsL[0],hsR[0])
    end = min(hsL[-1],hsR[-1])
    data = datas[int(start2-WindowSize+1):int(end+1)]
    
    #윈도우
    GP = Gait_label(hsL,hsR)
    
    X,Y = window(WindowSize,1,data,GP)

    #testset 분리
    if trial < 3 :
        x_test_n = np.concatenate((x_test_n,X),axis=2)
        y_test_n = np.concatenate((y_test_n,Y),axis=1)
    else :
        x_train_n = np.concatenate((x_train_n,X),axis=2)
        y_train_n = np.concatenate((y_train_n,Y),axis=1)
        
x_test_n = np.delete(x_test_n,0,axis=2)
y_test_n = np.delete(y_test_n,0,axis=1)
x_train_n = np.delete(x_train_n,0,axis=2)
y_train_n = np.delete(y_train_n,0,axis=1)   

#%% Read csv slalom data

x_test_o = np.zeros((30,12,1))
x_train_o = np.zeros((30,12,1))
y_test_o = np.zeros((4,1))
y_train_o = np.zeros((4,1))

gap_list = [ '1','15','2']
con_list = [ '100','150','200']
#sub_list = [ '1','2','3','4','5','6','7','8','9','10']
sub_list = [ '3']
trial_list = {'1': {'100':25,'150':20,'200':15},'15': {'100':25,'150':20,'200':15},'2': {'100':25,'150':23,'200':28}}
accept_train_list = {'1': {'100':1,'150':1,'200':1},'15': {'100':1,'150':1,'200':1},'2': {'100':1,'150':1,'200':1}}

for gap_num in gap_list:
    for con_num in con_list:
        
        for i in range(trial_list[gap_num][con_num]):
            datas = pd.read_csv(Path+gap_num+'_'+con_num+'_sub_3_trial'+str(i+1)+'.csv')
            hsL = pd.read_csv(Path+'hs_l_'+gap_num+'_'+con_num+'_sub_3_trial'+str(i+1)+'.csv')
            hsR = pd.read_csv(Path+'hs_r_'+gap_num+'_'+con_num+'_sub_3_trial'+str(i+1)+'.csv')
            
            datas = datas.to_numpy()
            
            hsL = hsL.to_numpy().reshape(-1)
            hsR = hsR.to_numpy().reshape(-1)
            
            stridetimeL = np.diff(hsL)
            
            #hs기준으로 정리
            start1 = min(hsL[0],hsR[0])
            start2 = max(hsL[0],hsR[0])
            end = min(hsL[-1],hsR[-1])
            data = datas[int(start2-WindowSize+1):int(end+1)]
            
            #윈도우
            GP = Gait_label(hsL,hsR)
            
            X,Y = window(WindowSize,1,data,GP)

            #testset 분리
            if( i < 3 or accept_train_list[gap_num][con_num]==0):
                x_test_o = np.concatenate((x_test_o,X),axis=2)
                y_test_o = np.concatenate((y_test_o,Y),axis=1)
            else :
                x_train_o = np.concatenate((x_train_o,X),axis=2)
                y_train_o = np.concatenate((y_train_o,Y),axis=1)
                
x_test_o = np.delete(x_test_o,0,axis=2)
x_train_o = np.delete(x_train_o,0,axis=2)
y_test_o = np.delete(y_test_o,0,axis=1)
y_train_o = np.delete(y_train_o,0,axis=1)

#%% dataset
class Custom_dataset(Dataset):
    def __init__(self, Xdata, label, transform=None):
        self.Xdata = torch.FloatTensor(Xdata).to(device)
        self.labels = torch.FloatTensor(label).to(device)
        
        self.zero = np.shape(Xdata)[0]
        self.one = np.shape(Xdata)[1]
        self.two = np.shape(Xdata)[2]
        
        self.transform = transform
    
    def __len__(self):
        return self.two    
    
    def __getitem__(self,idx):
        datas = self.Xdata[:, :, idx].reshape((self.zero, self.one, 1))
        datas = torch.transpose(datas,0,2)
        #datas = datas.transpose(2,1)
        #datas = datas.unsqueeze(0)
        label = self.labels[:, idx].reshape(4)
        #label = label.transpose(1,0)
        return datas, label
    
#%% Net
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
    
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=(1,10) , stride=1, padding=(0,1)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
           )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 32, kernel_size=(12,5), stride=1, padding=(0,1)),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=(1,3), stride=(1,3))
            )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(224,  32, bias = True),
            torch.nn.ReLU(),
            torch.nn.Linear(32 ,  4))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC    
        out = self.layer3(out)
        
        return out
    
#%%
def RMSE(p,y):
    RMSE = np.sqrt((np.mean((p-y)**2)))/(np.max(y) - np.min(y))
    RMSE = round(RMSE,4)
    return RMSE

#%% create model
model = CNN().to(device)

model.load_state_dict(torch.load('./CNN_Net.pth', map_location=torch.device(device)))

loss_fn = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=0.003, weight_decay=1e-5) 


#%% create dataloader
test_dataset = Custom_dataset(np.concatenate((x_test_n,x_test_o),axis=2), np.concatenate((y_test_n,y_test_o),axis=1))
train_dataset = Custom_dataset(np.concatenate((x_train_n,x_train_o),axis=2), np.concatenate((y_train_n,y_train_o),axis=1))
trainloader = DataLoader(train_dataset, batch_size = 512 , shuffle= True , drop_last = True)
testloader = DataLoader(test_dataset, batch_size=1)

#%% train   
torch.backends.cudnn.enabled = False
batch_length = len(trainloader)
losses = []

model.train()

for epoch in range(10):
    avg_cost = 0
    epoch_loss = []
    for X, Y in trainloader:
        #X = X.view(64,12,1,30)
        #Y = Y.view(64,1,4)
        X = X.type(torch.FloatTensor).to(device)
        Y = Y.type(torch.FloatTensor).to(device)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = torch.sqrt(loss_fn(Y_pred, Y))
        epoch_loss.append(loss.data)
        loss.backward()
        optimizer.step()
        '''torch.sqrt(loss_fn(torch.atan2(Y[:,1],Y[:,0]),torch.atan2(Y_pred[:,1],Y_pred[:,0])) + loss_fn(torch.atan2(Y[:,3],Y[:,2]),torch.atan2(Y_pred[:,3],Y_pred[:,2])))'''
        avg_cost += loss / batch_length   
    #losses.append(avg_cost.item()) 
    losses.append(sum(epoch_loss)/len(epoch_loss)) 
    print("epoch : ",epoch," -- ", avg_cost)
 
torch.save(model.state_dict(), './CNN_Net.pth') 