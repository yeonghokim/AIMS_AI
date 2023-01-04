# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 16:30:42 2022

@author: 82109
"""

#%%
from __future__ import print_function
import torch
import scipy
import scipy.io
from scipy.stats import kstest
import pandas as pd
import numpy as np
import math
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import argparse
import os
import random
#from torchsummaryX import summary
#%%
torch.cuda.empty_cache()

seed = 1
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#torch.cuda.set_device(0)
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.cuda.manual_seed(seed)  # type: ignore
torch.backends.cudnn.deterministic = True  # type: ignore
torch.backends.cudnn.benchmark = True  # type: ignore
if device == 'cuda': torch.cuda.manual_seed_all(seed)
if device == 'cuda': torch.cuda.manual_seed_all(43)
#%% 
def sampling(imu):
    sampled = np.zeros((1,3))
    for i in range(0,len(imu),5):
        mean = np.mean(imu[i:i+5],axis=0)
        sampled = np.concatenate((sampled, [mean]), axis=0)  
    sampled = np.delete(sampled, [0,0], axis=0)    
    #sampled = np.delete(sampled, [0])  
    return sampled

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

#%%
def XY2Gait(estimation):
    phase_Est = np.zeros((2,estimation.shape[1]))
    
    thetaL = np.arctan2(estimation[1,:],estimation[0,:])
    sn = thetaL < 0
    thetaL = thetaL + (sn*math.pi*2)
    phase_Est[0,:] = thetaL/(2*math.pi)
    
    thetaR = np.arctan2(estimation[3,:],estimation[2,:])
    sn = thetaR < 0
    thetaR = thetaR + (sn*math.pi*2)    
    phase_Est[1,:] = thetaR/(2*math.pi)
                
    return phase_Est
#%%넘파이 수정
def window(windowsize,shiftsize,X,Y):
    X_out = np.zeros((windowsize,X.shape[1],X.shape[0]-windowsize+1))
    Y_out = np.zeros((Y.shape[0],Y.shape[1]))
    
    for i in range(0,X.shape[0]-windowsize+1,shiftsize):
        for j in range(X.shape[1]):
            X_out[:,j,i] = X[i:i+windowsize,j]
    for i in range(0,Y.shape[1],shiftsize):
        Y_out[:,i] = Y[:,i]
    #X_out = X_out.unsqueeze(1)
    
    return X_out, Y_out


#%%
gap_list = [ '1','2','15']
con_list = [ '100','150','200']
#sub_list = [ '1','2','3','4','5','6','7','8','9','10']
sub_list = [ '3']


#%%임시
"""
gap_list = [ '1']
con_list = [ '100']
#sub_list = [ '1','2','3','4','5','6','7','8','9','10']
sub_list = [ '1']
"""

#%%
x_test_n = np.zeros((30,12,1))
x_train_n = np.zeros((30,12,1))
y_test_n = np.zeros((4,1))
y_train_n = np.zeros((4,1))
x_test_n_list = []
y_test_n_list = []

for u in range(20):
    datas = pd.read_csv('C:/Users/csh/sub_3/sub_3_trial'+str(u+1)+'.csv')
    hsL = pd.read_csv('C:/Users/csh/sub_3//hs_l_sub_3_trial'+str(u+1)+'.csv')
    hsR = pd.read_csv('C:/Users/csh/sub_3//hs_r_sub_3_trial'+str(u+1)+'.csv')
    datas = datas.to_numpy()
    hsL = hsL.to_numpy().reshape(-1)
    hsR = hsR.to_numpy().reshape(-1)
    
    stridetimeL = np.diff(hsL)
    #print(stridetimeL)
    
    #hs기준으로 정리
    start1 = min(hsL[0],hsR[0])
    start2 = max(hsL[0],hsR[0])
    end = min(hsL[-1],hsR[-1])
    #data1 = datas[int(start1):int(end+1)]
    data = datas[int(start2-30+1):int(end+1)]
    
    #윈도우
    GP = Gait_label(hsL,hsR)
    
    #data = torch.FloatTensor(data2).to(device)
    #GP = torch.FloatTensor(GP).to(device)
    X,Y = window(30,1,data,GP)

    #testset 분리
    if u < 3 :
        x_test_n = np.concatenate((x_test_n,X),axis=2)
        y_test_n = np.concatenate((y_test_n,Y),axis=1)
    else :
        x_train_n = np.concatenate((x_train_n,X),axis=2)
        y_train_n = np.concatenate((y_train_n,Y),axis=1)
        
x_test_n = np.delete(x_test_n,0,axis=2)
y_test_n = np.delete(y_test_n,0,axis=1)
x_test_n_list.append(x_test_n)
y_test_n_list.append(y_test_n)
       # x_test_list = np.append(x_test_list, [x_test_])

x_train_n = np.delete(x_train_n,0,axis=2)
y_train_n = np.delete(y_train_n,0,axis=1)        
        
#%%

x_test_o = np.zeros((30,12,1))
x_train_o = np.zeros((30,12,1))
y_test_o = np.zeros((4,1))
y_train_o = np.zeros((4,1))
x_test_list = []
y_test_list = []
for gap_num in gap_list:
    for con_num in con_list:
        trial_title = "C:/Users/csh/gait/Slalom walking test_220405/"+gap_num+"m_"+con_num+"cm.mat"
        traindata = scipy.io.loadmat(trial_title)
        x_test_ = np.zeros((30,12,1))
        y_test_ = np.zeros((4,1))
        for sub_num in sub_list:
            
            aR_Y = np.zeros((1))
            angleR_Y =traindata['Angle_'+gap_num+'m_'+con_num+'cm_right_sub'+sub_num][:,0]
            aR_Y = np.concatenate((aR_Y,angleR_Y), axis=0)
        
        aR_Y = np.delete(aR_Y,0,axis=0)
        x = aR_Y
        peaks, _ = find_peaks(x, distance=800)
        for i in range(len(peaks)-1):
        #for i in range(1):
            datas = pd.read_csv('C:/Users/csh/sub_3/'+gap_num+'_'+con_num+'_sub_3_trial'+str(i+1)+'.csv')
            hsL = pd.read_csv('C:/Users/csh/sub_3/'+'hs_l_'+gap_num+'_'+con_num+'_sub_3_trial'+str(i+1)+'.csv')
            hsR = pd.read_csv('C:/Users/csh/sub_3/'+'hs_r_'+gap_num+'_'+con_num+'_sub_3_trial'+str(i+1)+'.csv')
            datas = datas.to_numpy()
            hsL = hsL.to_numpy().reshape(-1)
            hsR = hsR.to_numpy().reshape(-1)
            
            stridetimeL = np.diff(hsL)
            #print(stridetimeL)
            
            #hs기준으로 정리
            start1 = min(hsL[0],hsR[0])
            start2 = max(hsL[0],hsR[0])
            end = min(hsL[-1],hsR[-1])
            #data1 = datas[int(start1):int(end+1)]
            data = datas[int(start2-30+1):int(end+1)]
            
            #윈도우
            GP = Gait_label(hsL,hsR)
            
            #data = torch.FloatTensor(data2).to(device)
            #GP = torch.FloatTensor(GP).to(device)
            X,Y = window(30,1,data,GP)

            #testset 분리
            if i < 3 :
                x_test_o = np.concatenate((x_test_o,X),axis=2)
                x_test_ = np.concatenate((x_test_,X),axis=2)
                y_test_ = np.concatenate((y_test_,Y),axis=1)
                y_test_o = np.concatenate((y_test_o,Y),axis=1)
            else :
                x_train_o = np.concatenate((x_train_o,X),axis=2)
                y_train_o = np.concatenate((y_train_o,Y),axis=1)
                
        x_test_ = np.delete(x_test_,0,axis=2)
        y_test_ = np.delete(y_test_,0,axis=1)
        x_test_list.append(x_test_)
        y_test_list.append(y_test_)
       # x_test_list = np.append(x_test_list, [x_test_])
x_test_o = np.delete(x_test_o,0,axis=2)
x_train_o = np.delete(x_train_o,0,axis=2)
y_test_o = np.delete(y_test_o,0,axis=1)
y_train_o = np.delete(y_train_o,0,axis=1)


#%% testset
#x_train = x_train_o[:,[0,1,2,3,4,5,6,7,8,9,10,11],:]
x_train = np.concatenate((x_train_o,x_train_n),axis=2)
x_train = x_train[:,[0,1,2,3,4,5,6,7,8,9,10,11],:]
x_test = np.concatenate((x_test_o,x_test_n),axis=2)

x_test_1_100 = x_test_list[0]
x_test_1_150 = x_test_list[1]
x_test_1_200 = x_test_list[2]
x_test_2_100 = x_test_list[3]
x_test_2_150 = x_test_list[4]
x_test_2_200 = x_test_list[5]
x_test_15_100 = x_test_list[6]
x_test_15_150 = x_test_list[7]
x_test_15_200 = x_test_list[8]

x_test_1_100 = x_test_1_100[:,[0,1,2,3,4,5,6,7,8,9,10,11],:]
x_test_1_150 = x_test_1_150[:,[0,1,2,3,4,5,6,7,8,9,10,11],:]
x_test_1_200 = x_test_1_200[:,[0,1,2,3,4,5,6,7,8,9,10,11],:]
x_test_2_100 = x_test_2_100[:,[0,1,2,3,4,5,6,7,8,9,10,11],:]
x_test_2_150 = x_test_2_150[:,[0,1,2,3,4,5,6,7,8,9,10,11],:]
x_test_2_200 = x_test_2_200[:,[0,1,2,3,4,5,6,7,8,9,10,11],:]
x_test_15_100 = x_test_15_100[:,[0,1,2,3,4,5,6,7,8,9,10,11],:]
x_test_15_150 = x_test_15_150[:,[0,1,2,3,4,5,6,7,8,9,10,11],:]
x_test_15_200 = x_test_15_200[:,[0,1,2,3,4,5,6,7,8,9,10,11],:]
x_test = x_test[:,[0,1,2,3,4,5,6,7,8,9,10,11],:]
x_test_n = x_test_n[:,[0,1,2,3,4,5,6,7,8,9,10,11],:]

y_test_1_100 = y_test_list[0]
y_test_1_150 = y_test_list[1]
y_test_1_200 = y_test_list[2]
y_test_2_100 = y_test_list[3]
y_test_2_150 = y_test_list[4]
y_test_2_200 = y_test_list[5]
y_test_15_100 = y_test_list[6]
y_test_15_150 = y_test_list[7]
y_test_15_200 = y_test_list[8]
 
#x_test_1_100 = torch.FloatTensor(x_test_1_100).to(device)
x_test_1_100 = torch.FloatTensor(x_test_1_100).to(device)
x_test_1_150 = torch.FloatTensor(x_test_1_150).to(device)
x_test_1_200 = torch.FloatTensor(x_test_1_200).to(device)
x_test_2_100 = torch.FloatTensor(x_test_2_100).to(device)
x_test_2_150 = torch.FloatTensor(x_test_2_150).to(device)
x_test_2_200 = torch.FloatTensor(x_test_2_200).to(device)
x_test_15_100 = torch.FloatTensor(x_test_15_100).to(device)
x_test_15_150 = torch.FloatTensor(x_test_15_150).to(device)
x_test_15_200 = torch.FloatTensor(x_test_15_200).to(device)
x_test_n = torch.FloatTensor(x_test_n).to(device) 

#xte1100 = x_test_1_150.numpy()[:,:,:100] 
#x_test = x_test[:,[0,1,2,3,4,5,6,7,8,9,10,11],:]

x_test = torch.FloatTensor(x_test).to(device)
x_train = torch.FloatTensor(x_train).to(device)

y_test = np.concatenate((y_test_o,y_test_n),axis=1)
y_train = np.concatenate((y_train_o,y_train_n),axis=1)

y_test_1_100 = torch.FloatTensor(y_test_1_100).to(device)
y_test_1_150 = torch.FloatTensor(y_test_1_150).to(device)
y_test_1_200 = torch.FloatTensor(y_test_1_200).to(device)
y_test_2_100 = torch.FloatTensor(y_test_2_100).to(device)
y_test_2_150 = torch.FloatTensor(y_test_2_150).to(device)
y_test_2_200 = torch.FloatTensor(y_test_2_200).to(device)
y_test_15_100 = torch.FloatTensor(y_test_15_100).to(device)
y_test_15_150 = torch.FloatTensor(y_test_15_150).to(device)
y_test_15_200 = torch.FloatTensor(y_test_15_200).to(device)
y_test_n = torch.FloatTensor(y_test_n).to(device) 


y_test = torch.FloatTensor(y_test).to(device)
y_train = torch.FloatTensor(y_train).to(device)
#x_test = x_test.unsqueeze(1)
#x_train = x_train.unsqueeze(1)
#%% dataset 임시
class Custom_dataset(Dataset):
    def __init__(self, Xdata, label, transform=None):
        self.Xdata = Xdata
        self.labels = label
        
        self.zero = np.shape(Xdata)[0]
        self.one = np.shape(Xdata)[1]
        self.two = np.shape(Xdata)[2]
        
        self.transform = transform
    
    def __len__(self):
        return self.two    
    
    def __getitem__(self,idx):
        datas = self.Xdata[:, :, idx].reshape((self.zero, self.one, 1))
        datas = datas.transpose(2,0)
        #datas = datas.transpose(2,1)
        #datas = datas.unsqueeze(0)
        label = self.labels[:, idx].reshape(4)
        #label = label.transpose(1,0)
        return datas, label

#%%임시 kernal 12,5fh
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

#%%
model = CNN().to(device)
#loss_fn = torch.nn.L1Loss().to(device)
loss_fn = torch.nn.MSELoss().to(device)
#optimizer = torch.optim.SGD(model.parameters(),lr=0.03, weight_decay=1e-5) 
optimizer = torch.optim.Adam(model.parameters(),lr=0.003, weight_decay=1e-5) 
#%%
test_dataset = Custom_dataset(x_test, y_test)
train_dataset = Custom_dataset(x_train, y_train)
trainloader = DataLoader(train_dataset, batch_size = 512 , shuffle= True , drop_last = True)
testloader = DataLoader(test_dataset, batch_size=1)

#%% train   
torch.backends.cudnn.enabled = False
batch_length = len(trainloader)
losses = []
model.train()
epochs = 300

for epoch in range(epochs):
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
    if epoch % 5 == 0 : print("epoch : ",epoch," -- ", avg_cost)
 
torch.save(model.state_dict(), 'C:/Users/csh/model_sub_3/2layer_12x3_1x10_seed1.pt')             
#%%            
def detach(Q,L):
    Q = Q.to(torch.device("cpu"))
    Q = Q.detach().numpy()
    
    L = L.to(torch.device("cpu"))
    L = L.detach().numpy()
    
    return Q,L
#%%
def Gait(Q,L):
    Q = XY2Gait(Q)
    L = XY2Gait(L)
    
    return Q,L

#%%
ind = ["C1","C1.5","C2"]
col = ["G1","G1.5","G2",'Normal']
R = pd.DataFrame(0,col,ind)

#%%

test_dataset_1_100 = Custom_dataset(x_test_1_100, y_test_1_100)
test_dataset_1_150 = Custom_dataset(x_test_1_150, y_test_1_150)
test_dataset_1_200 = Custom_dataset(x_test_1_200, y_test_1_200)
test_dataset_15_100 = Custom_dataset(x_test_15_100, y_test_15_100)
test_dataset_15_150 = Custom_dataset(x_test_15_150, y_test_15_150)
test_dataset_15_200 = Custom_dataset(x_test_15_200, y_test_15_200)
test_dataset_2_100 = Custom_dataset(x_test_2_100, y_test_2_100)
test_dataset_2_150 = Custom_dataset(x_test_2_150, y_test_2_150)
test_dataset_2_200 = Custom_dataset(x_test_2_200, y_test_2_200)
test_dataset_n = Custom_dataset(x_test_n,y_test_n)

testloader_1_100 = DataLoader(test_dataset_1_100, batch_size=1)

testloader_1_150 = DataLoader(test_dataset_1_150, batch_size=1)
testloader_1_200 = DataLoader(test_dataset_1_200, batch_size=1)
testloader_15_100 = DataLoader(test_dataset_15_100, batch_size=1)
testloader_15_150 = DataLoader(test_dataset_15_150, batch_size=1)
testloader_15_200 = DataLoader(test_dataset_15_200, batch_size=1)
testloader_2_100 = DataLoader(test_dataset_2_100, batch_size=1)
testloader_2_150 = DataLoader(test_dataset_2_150, batch_size=1)
testloader_2_200 = DataLoader(test_dataset_2_200, batch_size=1)
testloader_n = DataLoader(test_dataset_n, batch_size=1)
#%%
y_test_1_100 = y_test_1_100.cpu().numpy()

y_test_1_150 = y_test_1_150.cpu().numpy()
y_test_1_200 = y_test_1_200.cpu().numpy()
y_test_15_100 = y_test_15_100.cpu().numpy()
y_test_15_150 = y_test_15_150.cpu().numpy()
y_test_15_200 = y_test_15_200.cpu().numpy()
y_test_2_100 = y_test_2_100.cpu().numpy()
y_test_2_150 = y_test_2_150.cpu().numpy()
y_test_2_200 = y_test_2_200.cpu().numpy()
y_test_n = y_test_n.cpu().numpy()
#%%
y_test = y_test.cpu().numpy()
#%%
correct = 0
total = 0

model.eval()

#testlist1 = [testloader_1_100, testloader_1_150, testloader_1_200]
with torch.no_grad():
    val_losses = []
    val_loss = 0.0
    outputlist = np.zeros((4,1))
 
    # 임시
    for data in testloader:
        images, labels = data
        outputs = model(images)
        v_loss = loss_fn(outputs, labels)
        val_loss += v_loss.data
        outputlist = np.concatenate((outputlist,outputs.cpu().T),axis=1)
    outputlist = np.delete(outputlist,0,axis=1)
    val_losses.append(val_loss)
    R.iloc[0,0] = RMSE(outputlist,y_test)
    Gait_pred_1_100, Gait_test_1_100 = Gait(outputlist , y_test)


    outputlist = np.zeros((4,1))
    
    for data in testloader_1_100:
        images, labels = data
        outputs = model(images)
        v_loss = loss_fn(outputs, labels)
        val_loss += v_loss.data
        outputlist = np.concatenate((outputlist,outputs.cpu().T),axis=1)
    outputlist = np.delete(outputlist,0,axis=1)
    val_losses.append(val_loss)
    R.iloc[0,0] = RMSE(outputlist,y_test_1_100)
    Gait_pred_1_100, Gait_test_1_100 = Gait(outputlist , y_test_1_100)

    outputlist = np.zeros((4,1))
    for data in testloader_1_150:
        images, labels = data
        outputs = model(images)
        outputlist = np.concatenate((outputlist,outputs.cpu().T),axis=1)
    outputlist = np.delete(outputlist,0,axis=1)
    R.iloc[0,1] = RMSE(outputlist,y_test_1_150)
    Gait_pred_1_150, Gait_test_1_150 = Gait(outputlist , y_test_1_150)
    
    outputlist = np.zeros((4,1))
    for data in testloader_1_200:
        images, labels = data
        outputs = model(images)
        outputlist = np.concatenate((outputlist,outputs.cpu().T),axis=1)
        _, predicted = torch.max(outputs, 1)
    outputlist = np.delete(outputlist,0,axis=1)
    R.iloc[0,2] = RMSE(outputlist,y_test_1_200)
    Gait_pred_1_200, Gait_test_1_200 = Gait(outputlist , y_test_1_200)
    
    outputlist = np.zeros((4,1))
    for data in testloader_15_100:
        images, labels = data
        outputs = model(images)
        outputlist = np.concatenate((outputlist,outputs.cpu().T),axis=1)
        _, predicted = torch.max(outputs, 1)
    outputlist = np.delete(outputlist,0,axis=1)
    R.iloc[1,0] = RMSE(outputlist,y_test_15_100)
    Gait_pred_15_100, Gait_test_15_100 = Gait(outputlist , y_test_15_100)
    
    outputlist = np.zeros((4,1))
    for data in testloader_15_150:
        images, labels = data
        outputs = model(images)
        outputlist = np.concatenate((outputlist,outputs.cpu().T),axis=1)
        _, predicted = torch.max(outputs, 1)
    outputlist = np.delete(outputlist,0,axis=1)
    R.iloc[1,1] = RMSE(outputlist,y_test_15_150)
    Gait_pred_15_150, Gait_test_15_150 = Gait(outputlist , y_test_15_150)
    
    outputlist = np.zeros((4,1))
    for data in testloader_15_200:
        images, labels = data
        outputs = model(images)
        outputlist = np.concatenate((outputlist,outputs.cpu().T),axis=1)
        _, predicted = torch.max(outputs, 1)
    outputlist = np.delete(outputlist,0,axis=1)
    R.iloc[1,2] = RMSE(outputlist,y_test_15_200)
    Gait_pred_15_200, Gait_test_15_200 = Gait(outputlist , y_test_15_200)
    
    outputlist = np.zeros((4,1))
    for data in testloader_2_100:
        images, labels = data
        outputs = model(images)
        outputlist = np.concatenate((outputlist,outputs.cpu().T),axis=1)
        _, predicted = torch.max(outputs, 1)
    outputlist = np.delete(outputlist,0,axis=1)
    R.iloc[2,0] = RMSE(outputlist,y_test_2_100)
    Gait_pred_2_100, Gait_test_2_100 = Gait(outputlist , y_test_2_100)
    
    outputlist = np.zeros((4,1))
    for data in testloader_2_150:
        images, labels = data
        outputs = model(images)
        outputlist = np.concatenate((outputlist,outputs.cpu().T),axis=1)
        _, predicted = torch.max(outputs, 1)
    outputlist = np.delete(outputlist,0,axis=1)
    R.iloc[2,1] = RMSE(outputlist,y_test_2_150)
    Gait_pred_2_150, Gait_test_2_150 = Gait(outputlist , y_test_2_150)
    
    outputlist = np.zeros((4,1))
    for data in testloader_2_200:
        images, labels = data
        outputs = model(images)
        outputlist = np.concatenate((outputlist,outputs.cpu().T),axis=1)
        _, predicted = torch.max(outputs, 1)
    outputlist = np.delete(outputlist,0,axis=1)
    R.iloc[2,2] = RMSE(outputlist,y_test_2_200)
    Gait_pred_2_200, Gait_test_2_200 = Gait(outputlist , y_test_2_200)
    
    outputlist = np.zeros((4,1))
    for data in testloader_n:
        images, labels = data
        outputs = model(images)
        outputlist = np.concatenate((outputlist,outputs.cpu().T),axis=1)
        _, predicted = torch.max(outputs, 1)
    outputlist = np.delete(outputlist,0,axis=1)
    R.iloc[3,0] = RMSE(outputlist,y_test_n)
    Gait_pred_n, Gait_test_n = Gait(outputlist , y_test_n)
    
        #total += labels.size(0)
        #correct += (predicted == labels).sum().item()

#print('Accuracy of the network on the 10000 test images: %d %%' % (
      #100 * correct / total))      

#R.to_excel('C:/Users/csh/R2.xlsx')

#%%
plt.figure(figsize=(16,16))
plt.title("Loss with window 30")
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.plot(losses)
"""
#%% 변환 plot
GP1100 = Gait_pred_1_100[0].T
GT1100 = Gait_test_1_100[0].T
plt.figure(figsize=(20,5))
plt.title("gait plot")
plt.xlabel('T(s)')
plt.ylabel('Gait Phase')
plt.plot(GP1100,label="est")
plt.plot(GT1100,label="test") 
#%% 변환 plot
GP1150 = Gait_pred_1_150[0].T
GT1150 = Gait_test_1_150[0].T
plt.figure(figsize=(16,16))
plt.title("gait plot")
plt.ylabel('Gait Phase')
plt.plot(GP1150[:500],label="est")
plt.plot(GT1150[:500],label="test")
#%% 변환 plot
GP1200 = Gait_pred_1_200[0].T
GT1200 = Gait_test_1_200[0].T
plt.figure(figsize=(16,16))
plt.title("gait plot")
plt.ylabel('Gait Phase')
plt.plot(GP1200[:500],label="est")
plt.plot(GT1200[:500],label="test")
#%% 변환 plot
GP2100 = Gait_pred_2_100[0].T
GT2100 = Gait_test_2_100[0].T
plt.figure(figsize=(16,16))
plt.title("gait plot")
plt.ylabel('Gait Phase')
plt.plot(GP2100[:500],label="est")
plt.plot(GT2100[:500],label="test")
#%% 변환 plot
GP2150 = Gait_pred_2_150[0].T
GT2150 = Gait_test_2_150[0].T
plt.figure(figsize=(20,5))
plt.title("gait plot")
plt.ylabel('Gait Phase')
plt.plot(GP2150,label="est")
plt.plot(GT2150,label="test")
#%% 변환 plot
GP2200 = Gait_pred_2_200[0].T
GT2200 = Gait_test_2_200[0].T
plt.figure(figsize=(16,16))
plt.title("gait plot")
plt.ylabel('Gait Phase')
plt.plot(GP2200[:500],label="est")
plt.plot(GT2200[:500],label="test")
#%% 변환 plot
GP15100 = Gait_pred_15_100[0].T
GT15100 = Gait_test_15_100[0].T
plt.figure(figsize=(16,16))
plt.title("gait plot")
plt.ylabel('Gait Phase')
plt.plot(GP15100[:500],label="est")
plt.plot(GT15100[:500],label="test")
#%% 변환 plot
GP15150 = Gait_pred_15_150[0].T
GT15150 = Gait_test_15_150[0].T
plt.figure(figsize=(16,16))
plt.title("gait plot")
plt.ylabel('Gait Phase')
plt.plot(GP15150[:500],label="est")
plt.plot(GT15150[:500],label="test")
#%% 변환 plot
GP15200 = Gait_pred_15_200[0].T
GT15200 = Gait_test_15_200[0].T
plt.figure(figsize=(16,16))
plt.title("gait plot")
plt.ylabel('Gait Phase')
plt.plot(GP15200[:500],label="est")
plt.plot(GT15200[:500],label="test")
#%% 변환 plot
GPn = Gait_pred_n[0].T
GTn = Gait_test_n[0].T
plt.figure(figsize=(16,16))
plt.title("gait plot")
plt.ylabel('Gait Phase')
plt.plot(GPn[:500],label="est")
plt.plot(GTn[:500],label="test")
"""