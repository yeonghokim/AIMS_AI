# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 20:14:03 2022

@author: yeong
"""
import math
import numpy as np
import platform
Linux_filepath = "/data/yeongho/task1/data/sub1"

Window_filepath = "C:/Users/yeong/Desktop/ResearchStudent/Github/AIMS_AI/data/Task2/sub1"
filepath=''
if platform.system()=='Windows':
    filepath = Window_filepath
    print('Windows filepath')
elif platform.system()=='Linux':
    filepath = Linux_filepath
    print('Linux filepath')
else:
    exit()
#%% Parameter
CNNinput_width=100

#%% file read
ex_type= ["W","R"]
ex_speed = ["fast","mid","slow"]
ex_feet = ["L","R"]
ex_data = ["angle","HS"]

Traindata_L_Angle=list() #input
Traindata_L_HS =list() #output

Traindata_R_Angle=list() #input
Traindata_R_HS =list() #output
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
                            
#%% Input data 
import Gait_label as gl


Traindata = np.empty((0,2,100), float)
Trainlabel = np.empty((0,4), float)

Testdata = np.empty((0,2,100), float)
Testlabel = np.empty((0,4), float)
for i in range(0,10):
    LHS = np.array(Traindata_L_HS[i])
    LAG = np.array(Traindata_L_Angle[i])
    RHS = np.array(Traindata_R_HS[i])
    RAG = np.array(Traindata_R_Angle[i])
    HSPhase = gl.Gait_label(LHS,RHS)
    
    for flag in range(HSPhase[0],len(HSPhase[1])):
        if flag<CNNinput_width: continue
    
        AG = np.stack((LAG[flag-CNNinput_width:flag],RAG[flag-CNNinput_width:flag]))
        if i<7:
            Traindata=np.insert(Traindata,len(Traindata),AG,axis=0)
            Trainlabel=np.insert(Trainlabel,len(Trainlabel),[HSPhase[1][flag]],axis=0)
        else:
            Testdata=np.insert(Testdata,len(Traindata),AG,axis=0)
            Testlabel=np.insert(Testlabel,len(Trainlabel),[HSPhase[1][flag]],axis=0)

#%% csv Traindata
f = open("Traindata.csv", "w")
for i in range(0,len(Traindata)):
    array = np.reshape(Traindata[i],(1,200))[0]
    for j in range(0,200):
        f.write(str(array[j])+' ')
    f.write('\n')
f.close()
    
#%% csv Trainlabel
f = open("Trainlabel.csv", "w")
for i in range(0,len(Trainlabel)):
    array =Trainlabel[i]
    for j in range(0,4):
        f.write(str(array[j])+' ')
    f.write('\n')
f.close()

#%% csv Testdata
f = open("Testdata.csv", "w")
for i in range(0,len(Testdata)):
    array = np.reshape(Testdata[i],(1,200))[0]
    for j in range(0,200):
        f.write(str(array[j])+' ')
    f.write('\n')
f.close()

#%% csv Testlabel
f = open("Testlabel.csv", "w")
for i in range(0,len(Testlabel)):
    array =Testlabel[i]
    for j in range(0,4):
        f.write(str(array[j])+' ')
    f.write('\n')
f.close()


