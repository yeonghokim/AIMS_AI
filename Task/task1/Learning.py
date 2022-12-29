# -*- coding: utf-8 -*-
import scipy.io
import numpy as np
import torch
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset


mat_file_name = "C:/Users/yeong/Desktop/학부연구생/task/sEMG(9).mat"
mat_file = scipy.io.loadmat(mat_file_name)

emg= mat_file['emg']
label  = mat_file['label']
rep = mat_file['repetition']

#Dictionary 로 각 label 과 repetition에 해당하는 신호 저장
allData = dict()
flag=0 
while flag < len(label) :
    if(label[flag,0]!=0):
        startflag=flag
        labelChar = 'L'+np.array2string(label[flag,0])+'-'+np.array2string(rep[flag,0])
        while label[flag,0]!=0 :
            flag+=1
        
        temp = emg[startflag:flag,:]
        allData.update({labelChar:temp})               
    
    flag+=1


EMGTrainValueList = list();
EMGTrainLabelList = list();

EMGTestValueList = list();
EMGTestLabelList = list();

#각 Segment들의 통계적 수치 구하기

allSegmentValueData=dict()
SegmentNumber=400
for key in allData:
    arr = allData[key]
    segNum =len(arr)//SegmentNumber
    arr = arr[:SegmentNumber*segNum]
    arr=np.split(arr,segNum,axis=0)
    
    valuearr = np.ones((segNum,12,3),dtype=np.float32)
    segNum=0;
    for segArr in arr:
        for segChannel in range(0,12):
            MAV = (np.abs(segArr[:,segChannel])).sum()/SegmentNumber
            
            VAR = (segArr[:,segChannel]**2).sum()/(SegmentNumber-1)
            
            tmp = segArr[1:,segChannel]-segArr[:(SegmentNumber-1),segChannel]
            WL = (np.abs(tmp)).sum()
            
            valuearr[segNum][segChannel] = [MAV,VAR,WL]
        
        segNum+=1
    allSegmentValueData[key] =valuearr
    for tmp in valuearr :
        if int(key[-1])>4 :         
            EMGTestValueList.append(tmp)
            if key[3]=='-' :
                EMGTestLabelList.append(int(key[1:3])-1)
            else:
                EMGTestLabelList.append(int(key[1])-1)
        else:
            EMGTrainValueList.append(tmp)
            if key[3]=='-' :
                EMGTrainLabelList.append(int(key[1:3])-1)      
            else:
                EMGTrainLabelList.append(int(key[1])-1)

from matplotlib import pyplot as plt

y= np.array(allSegmentValueData['L16-1'])

#y= np.array(allData['L1-1'])
#for i in range(0,12):
#    plt.plot(np.arange(0,y.shape[0],1),y[:,i],label='Ch'+str(i+1))
#plt.legend(loc='upper right')
#plt.xlabel('Channel')
#plt.ylabel('EMG')
#plt.title('L1-1 EMG Graph')
#plt.show()


for i in range(0,16):
    plt.plot(np.arange(1,13,1),y[i,:,0])
#plt.legend(loc='upper right')
plt.xlabel('Channel')
plt.ylabel('EMG')
plt.title('L16-1 MAV Graph')
channellabel = list(map(str, range(1,13)))
for i in range(0,12):
    channellabel[i] = 'ch'+channellabel[i]
plt.xticks(range(1,13), channellabel)
plt.show()



#0.5ms 초마다 한번 체크한것이므로 200ms만큼 나눌려면 400개 필요
class MyEMGData(Dataset):
    def __init__(self,emgData,label):
        #리스트로 다 들어옴
        self.emgData = torch.FloatTensor(emgData)
        self.label = torch.LongTensor(label)
        self.len = self.label.shape[0]
        
    def __getitem__(self,index):
        return self.emgData[index], self.label[index]
    
    def __len__(self):
        return self.len
    

class ToTensor:
    def __call__(self,sample):
        emg, label = sample
        return torch.FloatTensor(emg), torch.LongTensor(label)
    
class LinearTensor:
    def __init__(self,slope=1,bias=0):
        self.slope = slope
        self.bias = bias
        
    def __call__(self,sample):
        emg, label = sample
        emg = self.slope*emg+self.bias
        return emg, label
    
    """
trans = tr.Compose([ToTensor(),LinearTensor(2,5)])
emg1 = MyEMGData(allSegmentValueData['L1-1'],1,transform=trans)
"""
train_dataset = MyEMGData(np.array(EMGTrainValueList),np.array(EMGTrainLabelList))
train_dataloader = DataLoader(train_dataset,batch_size=10,shuffle=True)

test_dataset = MyEMGData(np.array(EMGTestValueList),np.array(EMGTestLabelList))
test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=True)

dataiter= iter(train_dataloader)
emgData, label = dataiter.next()
print(emgData.size())
#rep 1~4 -> train set 5~6 test set

import torch.nn as nn
import torch.nn.functional as F
    
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(36,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,512)
        self.fc4 = nn.Linear(512,512)
        self.fc5 = nn.Linear(512,512)
        self.fc6 = nn.Linear(512,512)
        self.fc7 = nn.Linear(512,17)
        
        self.dropout = nn.Dropout(0.25)
        # 12 X 3 -> 17 X 1

    def forward(self,x):
        x=x.view(-1,36)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x = self.dropout(x)
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        x=F.relu(self.fc6(x))
        x=self.fc7(x)
        return x
        
net = Net()

PATH = './MLP_net_layer7_dropout.pth'
net.load_state_dict(torch.load(PATH))

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.001,momentum=0.9)

for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(train_dataloader,0):
        inputs, labels = data
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss +=loss.item()
        if i % 10 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch+1,i+1,running_loss/10))
            running_loss=0.0

correct = 0
total = 0
predict = torch.tensor([])
target = torch.tensor([])
print(correct)
with torch.no_grad():
    for data in test_dataloader:
        inputs, labels = data
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data,1)
        predict=torch.cat([predict, predicted+1],dim=0)
        target=torch.cat([target, labels+1],dim=0)
            
        total+=labels.size(0)
        correct += (predicted == labels).sum().item()
        
print('Accuracy of the network on the 741 test : %d %%' %(100*correct/total))

from sklearn.metrics import confusion_matrix     
import seaborn as sn
import pandas as pd

confusionmatrix = confusion_matrix(target,predict)
outputlabel=('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17')

df_cm = pd.DataFrame(confusionmatrix, index = [i for i in outputlabel], columns = [i for i in outputlabel])
plt.figure(figsize = (17,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output.png')

PATH = './MLP_net_layer7_dropout.pth'
torch.save(net.state_dict(),PATH)


