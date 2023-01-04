# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:04:39 2022

@author: 82109
"""

#%%
import torch
import scipy
import scipy.io
from scipy.stats import kstest
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#from torchsummaryX import summary
#%%
torch.cuda.empty_cache()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(43)
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
#%%

gap_list = [ '1','2','15']
con_list = [ '100','150','200']
#sub_list = [ '1','2','3','4','5','6','7','8','9','10']
sub_list = [ '7']

#%%
# save
aL_R = np.zeros((1))
aR_R = np.zeros((1))
aL_Y = np.zeros((1))
aL_P = np.zeros((1))
aR_P = np.zeros((1))
aL_HS = np.zeros((1))
aR_HS = np.zeros((1))
iL_Y = np.zeros((1))
iR_Y = np.zeros((1))
iL_P = np.zeros((1))
iR_P = np.zeros((1))
iL_R = np.zeros((1))
iR_R = np.zeros((1))
iL_HS = np.zeros((1))
iR_HS = np.zeros((1))

for gap_num in gap_list:
    for con_num in con_list:
        trial_title = "C:/Users/82109/Downloads/Slalom walking test_220405/"+gap_num+"m_"+con_num+"cm.mat"
        traindata = scipy.io.loadmat(trial_title)
         
        for sub_num in sub_list:
            
            aR_Y = np.zeros((1))
            train = np.zeros((1,12))
            
            angleL = traindata['Angle_'+gap_num+'m_'+con_num+'cm_left_sub'+sub_num][:,:]
            angleR =traindata['Angle_'+gap_num+'m_'+con_num+'cm_right_sub'+sub_num][:,:]
            angleL_P =traindata['Angle_'+gap_num+'m_'+con_num+'cm_left_sub'+sub_num][:,2]
            angleL_R =traindata['Angle_'+gap_num+'m_'+con_num+'cm_left_sub'+sub_num][:,1]
            angleL_Y =traindata['Angle_'+gap_num+'m_'+con_num+'cm_left_sub'+sub_num][:,0]
            angleR_P =traindata['Angle_'+gap_num+'m_'+con_num+'cm_right_sub'+sub_num][:,2]
            angleR_R =traindata['Angle_'+gap_num+'m_'+con_num+'cm_right_sub'+sub_num][:,1]
            angleR_Y =traindata['Angle_'+gap_num+'m_'+con_num+'cm_right_sub'+sub_num][:,0]
            aL_HS =traindata['HS_angle_'+gap_num+'m_'+con_num+'cm_left_sub'+sub_num][:,0]
            aR_HS =traindata['HS_angle_'+gap_num+'m_'+con_num+'cm_right_sub'+sub_num][:,0]
            imuL =traindata['IMU_'+gap_num+'m_'+con_num+'cm_left_sub'+sub_num][:,3:]
            imuR =traindata['IMU_'+gap_num+'m_'+con_num+'cm_right_sub'+sub_num][:,3:]
            imuL_HS =traindata['HS_IMU_'+gap_num+'m_'+con_num+'cm_left_sub'+sub_num][:,0]#.reshape(-1)
            imuR_HS =traindata['HS_IMU_'+gap_num+'m_'+con_num+'cm_right_sub'+sub_num][:,0]#.reshape(-1)
        
            # imu sampling
            imuL = sampling(imuL)
            imuR = sampling(imuR)
            imuL_Y = imuL[:,0]
            imuL_R = imuL[:,1]
            imuL_P = imuL[:,2]
            imuR_Y = imuL[:,0]
            imuR_R = imuL[:,1]
            imuR_P = imuL[:,2]
            
            angleL = angleL[:len(imuL),:]
            angleR = angleR[:len(imuR),:]
            
            tra = np.concatenate([angleL,angleR,imuL,imuR],axis=1)
            
            #aL_HS = np.concatenate((aL_HS,angleL_HS), axis=0)
            #aR_HS = np.concatenate((aR_HS,angleR_HS), axis=0)
            #iL_HS = np.concatenate((iL_HS,angleL_HS), axis=0)
            #iR_HS = np.concatenate((iR_HS,angleR_HS), axis=0)
        
            aR_Y = np.delete(aR_Y,0,axis=0)
            train = np.delete(train,0,axis=0)
            
            x = angleR_Y
            peaks, _ = find_peaks(x, distance=800)
            #print(peaks)
            
            for i in range(len(peaks)-1):
            #for i in range(1):
                trialdata = np.zeros((1,12))
                lhsdata = np.zeros((1))
                rhsdata = np.zeros((1))
                for k in range(len(aL_HS)):
                    if (aL_HS[k] > peaks[i]) and (aL_HS[k] < peaks[i+1]) :
                        Lhs = aL_HS[k]-peaks[i]
                        lhsdata = np.concatenate((lhsdata,[Lhs]),axis=0)
                lhsdata = np.delete(lhsdata,0,axis=0)
                df2 = pd.DataFrame(lhsdata)
                df2.to_csv('C:/Users/82109/sub_7/hs_l_'+gap_num+'_'+con_num+'_sub_7_trial'+str(i+1)+'.csv', index=False) 
                for m in range(len(aR_HS)):
                    if (aR_HS[m] > peaks[i]) and (aR_HS[m] < peaks[i+1]) :
                        Rhs = aR_HS[m]-peaks[i]
                        rhsdata = np.concatenate((rhsdata,[Rhs]),axis=0)
                rhsdata = np.delete(rhsdata,0,axis=0)
                df3 = pd.DataFrame(rhsdata)
                df3.to_csv('C:/Users/82109/sub_7/hs_r_'+gap_num+'_'+con_num+'_sub_7_trial'+str(i+1)+'.csv', index=False)
                for j in range(peaks[i],peaks[i+1]) :
                #for j in range(peaks[i],peaks[i]+2) :
                    tri = tra[j].reshape((1,12))
                    trialdata = np.concatenate((trialdata,tri), axis=0)
                trialdata = np.delete(trialdata,0,axis=0)
                df = pd.DataFrame(trialdata)
                df.to_csv('C:/Users/82109/sub_7/'+gap_num+'_'+con_num+'_sub_7_trial'+str(i+1)+'.csv', index=False) 
                