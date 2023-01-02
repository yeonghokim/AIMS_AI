#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:55:35 2022

@author: jaeyoung
"""


import math
import numpy as np
# import pandas as pd
import scipy.io
import copy
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter,lfilter,filtfilt

#%%
def Gait_label(HSR,HSL):     # input: data size, [HSR HSL]    output: R[x y] L[x y]

    #Left Right중 큰 것 찾기
    first_index = int(max(HSR[0],HSL[1]))
    
    #Left Right중 작은 것 찾기
    last_index = min(HSR[-1],HSL[-1])
    
    
    n = int(last_index)
    # print(n)
    true_phase = np.zeros((4,n))

    #각각 차이를 가져오기
    stridetimeL = np.diff(HSL)
    stridetimeR = np.diff(HSR)    


    phasedataR =np.zeros((int(HSR[0])-1))
    for i in range(len(stridetimeR)):
        pdata = np.linspace(0,1,int(stridetimeR[i]+1))
#        pdata = np.arange(0,1,1/stridetimeR[i])

        pdata = np.delete(pdata, len(pdata)-1,0)
        phasedataR = np.append(phasedataR,pdata)
#    phasedataR = np.delete(phasedataR, [0])
    phasedataR = np.append(phasedataR ,1)
    # print(len(phasedataR))
    phasedataR = phasedataR[:n]
    theta = phasedataR*2*math.pi
    x = np.cos(theta)
    y = np.sin(theta)
    true_phase[0,:] = x
    true_phase[1,:] = y
    
    phasedataL =np.zeros((int(HSL[0])-1))
    for i in range(len(stridetimeL)):
        pdata = np.linspace(0,1,int(stridetimeL[i]+1))
#        pdata = np.arange(0,1,1/stridetimeL[i])
        pdata = np.delete(pdata, len(pdata)-1,0)
        phasedataL = np.append(phasedataL,pdata)
#    phasedataL = np.delete(phasedataL,0,0)
    phasedataL = np.append(phasedataL ,1)
    # print(len(phasedataL))
    phasedataL = phasedataL[:n]
    
    theta = phasedataL*2*math.pi
    x = np.cos(theta)
    y = np.sin(theta)
    true_phase[2,:] = x
    true_phase[3,:] = y
    true_phase = np.transpose(true_phase)
    
    return first_index,true_phase
    
#%%
def XY2Gait(estimation, true):

    thetaL = np.arctan2(estimation[:, 1], estimation[:, 0])
    sn = thetaL < 0
    thetaL = thetaL + (sn * 1)*math.pi*2
    phase_EstL = thetaL*(1/(2*math.pi))

    theta_labelL = np.arctan2(true[:, 1], true[:, 0])
    sn = theta_labelL < 0
    theta_labelL = theta_labelL + (sn * 1)*math.pi*2
    phase_labelL = theta_labelL*(1/(2*math.pi))

    thetaR = np.arctan2(estimation[:, 3], estimation[:, 2])
    sn = thetaR < 0
    thetaR = thetaR + (sn * 1)*math.pi*2
    phase_EstR = thetaR*(1/(2*math.pi))

    theta_labelR = np.arctan2(true[:, 3], true[:, 2])
    sn = theta_labelR < 0
    theta_labelR = theta_labelR + (sn * 1)*math.pi*2
    phase_labelR = theta_labelR*(1/(2*math.pi))

    return phase_EstL, phase_labelL, phase_EstR, phase_labelR


#angle = to_roll_val(data_bR)
#first_index,gait = Gait_label(HS)
# def deletnan(data):
#     nan_list = np.argwhere(np.isnan(data))
#     if len(nan_list) == 0:
#         return data
#     else:
#         n = np.shape(nan_list)[0]
#         for i in range(n):
#             x,y = nan_list[i,:]
#             data[x,y] = (data[x-1,y]+data[x+1,y])/2
#         return data
    
def resample(data):
    num = int(len(data)/4)
    resampled_data = np.zeros((num))
    for i in range(num):
        # resampled_data[i] = (data[i*4]+data[i*4+1]+data[i*4+2]+data[i*4+3])/4    
        
        resampled_data[i] = data[i*4]
    return resampled_data

def data_scaler(window_data, time_data,angleminmax =None,angvleminmax =None,test=False): 
    scaled_window = copy.deepcopy(window_data)
    scaled_time = copy.deepcopy(time_data)
    angle_data = window_data[:,:,[0,2]]
    angvel_data =  window_data[:,:,[1,3]]
    scaled_angle_data = copy.deepcopy(angle_data)
    scaled_angvel_data =  copy.deepcopy(angvel_data)
    time_angle = time_data[:,[0,2]]
    time_angvel = time_data[:,[1,3]]
    angle_scaler = MinMaxScaler(feature_range=(0, 1),  copy=False)
    angvel_scaler = MinMaxScaler(feature_range=(0, 1),  copy=False)
    
    angle_scaler.fit(time_angle)
    angvel_scaler.fit(time_angvel)
    
    if test == True:
        angle_scaler.fit(angleminmax)
        angvel_scaler.fit(angvleminmax)
    angle_minmax = np.array( [angle_scaler.data_min_, angle_scaler.data_max_ ])
    angvel_minmax = np.array( [angvel_scaler.data_min_, angvel_scaler.data_max_ ])
    # print(angle_minmax)
    # scaled_angle_data = angle_data
    
    for i in range( np.shape(angle_data)[0]):
        scaled_angle_data[i,:,:] = angle_scaler.transform(scaled_angle_data[i,:,:])
        scaled_angvel_data[i,:,:] = angvel_scaler.transform(scaled_angvel_data[i,:,:])

    
    scaled_angle_time = angle_scaler.transform(time_angle)
    scaled_angvel_time = angvel_scaler.transform(time_angvel)

    scaled_window[:,:,0] = scaled_angle_data[:,:,0]
    scaled_window[:,:,1] = scaled_angvel_data[:,:,0]
    scaled_window[:,:,2] = scaled_angle_data[:,:,1]
    scaled_window[:,:,3] = scaled_angvel_data[:,:,1]

    scaled_time[:,0] = scaled_angle_time[:,0]
    scaled_time[:,1] = scaled_angvel_time[:,0]
    scaled_time[:,2] = scaled_angle_time[:,1]
    scaled_time[:,3] = scaled_angvel_time[:,1]
    return scaled_window,scaled_time ,angle_minmax, angvel_minmax

#%%
def make_dataset( window_size =100, stride =1,resampleshift=0, trial_list =['1','2','3'], Inclination_list = [ '00','01','02','03','04','05','06','07','08','09','10'], WS_list = ['1','2','3','4'],scaledata = False, train_true =True):
        
    # count = 1
# window_size = 100
# Inclination_list = [ '00','02','04','06','08','10']
# WS_list = ['1','2','3','4']

    # start_num = 100
    roll_gyro_list = np.zeros((1,window_size,4))
    roll_gyro_total_list= np.zeros((1,4))
    gait_label_list = np.zeros((1,4)) 
    incliation_list = np.zeros((1))
    walkingspeed_list = np.zeros((1))
    for trial_num in trial_list:
        roll_gyro_single_trial = np.zeros((1,window_size,4))
        roll_gyro_single_trial_list= np.zeros((1,4))
        gait_label_single_trial = np.zeros((1,4)) 
        incliation_single_trial = np.zeros((1))
        walkingspeed_single_trial = np.zeros((1))
        trial_title = './ascend_data/Day'+trial_num+'.mat'
        traindata = scipy.io.loadmat(trial_title)
        for inc in Inclination_list:
            if inc =='00':
                Inclination = 0
            elif inc =='01':
                Inclination = 0.1
            elif inc =='02':
                Inclination = 0.2
            elif inc =='03':
                Inclination = 0.3
            elif inc =='04':
                Inclination = 0.4
            elif inc =='05':
                Inclination = 0.5
            elif inc =='06':
                Inclination = 0.6
            elif inc =='07':
                Inclination = 0.7
            elif inc =='08':
                Inclination = 0.8
            elif inc =='09':
                Inclination = 0.9
            elif inc =='10':
                Inclination = 1
                
            
            for ws in WS_list:
                if (trial_num =='6') or( trial_num =='8'):
                    if ws =='1':
                        speed = 0.8
                    elif ws =='2':
                        speed = 1
                    elif ws =='3':
                        speed = 1.2
                else:
                    if ws =='1':
                        speed = 0.7
                    elif ws =='2':
                        speed = 0.9
                    elif ws =='3':
                        speed = 1.1
                    elif ws =='4':
                        speed = 1.3
                try:
                    if (trial_num == '4') & (inc == '00') & (ws =='1'):
                        trialangleR =traindata['angle_'+inc+'_'+ws+'_1_Angle_right'][:,0]
                        trialangleL =traindata['angle_'+inc+'_'+ws+'_1_Angle_left'][:,0]
                        trialanglevelR =traindata['angle_'+inc+'_'+ws+'_1_AngularVelocity_right'][:,0]
                        trialanglevelL =traindata['angle_'+inc+'_'+ws+'_1_AngularVelocity_left'][:,0]
                        trialHSR =traindata['angle_'+inc+'_'+ws+'_1_HS_right'].reshape(-1)
                        trialHSL =traindata['angle_'+inc+'_'+ws+'_1_HS_left'].reshape(-1)
                    else:
                        trialangleR =traindata['angle_'+inc+'_'+ws+'_Angle_right'][:,0]
                        trialangleL =traindata['angle_'+inc+'_'+ws+'_Angle_left'][:,0]
                        trialanglevelR =traindata['angle_'+inc+'_'+ws+'_AngularVelocity_right'][:,0]
                        trialanglevelL =traindata['angle_'+inc+'_'+ws+'_AngularVelocity_left'][:,0]
                        trialHSR =traindata['angle_'+inc+'_'+ws+'_HS_right'].reshape(-1)
                        trialHSL =traindata['angle_'+inc+'_'+ws+'_HS_left'].reshape(-1)
                    
                    resampled_angleR = resample(trialangleR)
                    resampled_angleL = resample(trialangleL)
                    resampled_anglevelR = resample(trialanglevelR)
                    resampled_anglrvelL = resample(trialanglevelL)
                    resampled_HSR = np.ceil( trialHSR/4)
                    resampled_HSL = np.ceil( trialHSL/4)
                    first_index,single_case_gait = Gait_label(resampled_HSR,resampled_HSL)
                    len_data = np.shape(single_case_gait)[0]
                    angleR_part = resampled_angleR[:len_data].reshape(-1,1)
                    angleL_part = resampled_angleL[:len_data].reshape(-1,1)
                    anglevelR_part = resampled_anglevelR[:len_data].reshape(-1,1)
                    anglevelL_part = resampled_anglrvelL[:len_data].reshape(-1,1)
                    
                    # print(np.shape(angleR_part),np.shape(angleL_part),np.shape(single_case_gait))
                    # tb_data = np.concatenate((angleR,angleL,gait),axis =1)
                    single_case_angle = np.concatenate((angleR_part,anglevelR_part,angleL_part,anglevelL_part),axis =1)
                    
                    # single_case_data = single_case_data[int(first_index):,:]
                    num_shift = int((len_data -first_index - window_size)/stride)
                    
                    roll_gyro_case = np.zeros((num_shift,4))
                    single_case_angle_part = np.zeros((num_shift,window_size,4))
                    single_case_gait_part = np.zeros((num_shift,4))
                    single_case_inclination_part =np.ones((num_shift))*Inclination
                    single_case_speed_part =np.ones((num_shift))*speed
                    
                    for j in range(num_shift):
                        
                        n = j*stride + first_index+resampleshift
                        single_case_angle_part[j,:,:] = single_case_angle[n:n+window_size,:]
                        single_case_gait_part[j,:] = single_case_gait[n+window_size-1,:]
                        roll_gyro_case[j,:] = single_case_angle[n+window_size-1,:]

                    split_num =-3000
                    if train_true == True:
                        split_num = int(split_num/stride)
                        roll_gyro_single_trial = np.concatenate((roll_gyro_single_trial,single_case_angle_part[:split_num,:,:]),axis=0)
                        roll_gyro_single_trial_list = np.concatenate((roll_gyro_single_trial_list,roll_gyro_case[:split_num,:]),axis=0)
                        gait_label_single_trial = np.concatenate((gait_label_single_trial,single_case_gait_part[:split_num,:]),axis=0)
                        incliation_single_trial = np.concatenate((incliation_single_trial,single_case_inclination_part[:split_num]),axis=0)
                        walkingspeed_single_trial = np.concatenate((walkingspeed_single_trial,single_case_speed_part[:split_num]),axis=0)
                    if train_true == False:
                        roll_gyro_single_trial = np.concatenate((roll_gyro_single_trial,single_case_angle_part[split_num:,:,:]),axis=0)
                        roll_gyro_single_trial_list = np.concatenate((roll_gyro_single_trial_list,roll_gyro_case[split_num:,:]),axis=0)
                        gait_label_single_trial = np.concatenate((gait_label_single_trial,single_case_gait_part[split_num:,:]),axis=0)
                        incliation_single_trial = np.concatenate((incliation_single_trial,single_case_inclination_part[split_num:]),axis=0)
                        walkingspeed_single_trial = np.concatenate((walkingspeed_single_trial,single_case_speed_part[split_num:]),axis=0)
                    # print(trial_num,inc,ws,num_shift ,count)
                    # count = count +1
                except:
                    continue

        
        roll_gyro_single_trial = np.delete(roll_gyro_single_trial,0,0)
        roll_gyro_single_trial_list = np.delete(roll_gyro_single_trial_list,0,0)
        # roll_gyro_total = np.delete(roll_gyro_total,0,0)
        gait_label_single_trial = np.delete(gait_label_single_trial,0,0)
        incliation_single_trial = np.delete(incliation_single_trial,0,0)
        walkingspeed_single_trial = np.delete(walkingspeed_single_trial,0,0)
        
        if scaledata ==True:
            if np.shape(roll_gyro_single_trial)[0] !=0:
                    
                roll_gyro_single_trial,roll_gyro_single_trial_list,angmin,angvelmin =data_scaler(roll_gyro_single_trial,roll_gyro_single_trial_list)
            # print(single_trial_angle_part[0,0,0])
        
        roll_gyro_list = np.concatenate((roll_gyro_list,roll_gyro_single_trial),axis=0)
        roll_gyro_total_list = np.concatenate((roll_gyro_total_list,roll_gyro_single_trial_list),axis=0)
        gait_label_list = np.concatenate((gait_label_list,gait_label_single_trial),axis=0)
        incliation_list = np.concatenate((incliation_list,incliation_single_trial),axis=0)
        walkingspeed_list = np.concatenate((walkingspeed_list,walkingspeed_single_trial),axis=0)
        
        # roll_gyro_single_trial = np.zeros((1,window_size,4))
        # roll_gyro_single_trial_list= np.zeros((1,4))
        # gait_label_single_trial = np.zeros((1,4)) 
        # incliation_single_trial = np.zeros((1))
        # walkingspeed_single_trial = np.zeros((1))
        
        
        print('trial ',trial_num,'dataset Done')
    roll_gyro_list = np.delete(roll_gyro_list,0,0)
    roll_gyro_total_list = np.delete(roll_gyro_total_list,0,0)
    # roll_gyro_total = np.delete(roll_gyro_total,0,0)
    gait_label_list = np.delete(gait_label_list,0,0)
    incliation_list = np.delete(incliation_list,0,0)
    walkingspeed_list = np.delete(walkingspeed_list,0,0)
    return roll_gyro_list ,roll_gyro_total_list, gait_label_list ,incliation_list , walkingspeed_list

 # test = False

def augment_shift(data,shift_list):
    window_data,time_data,gait_label, slopelabel, speedlabel = data
    l,w,h= np.shape(window_data)
    num = len(shift_list)
    total_window_data = np.zeros((1,w,h))
    total_time_data = np.tile(time_data,(num,1))
    total_gait_label = np.tile(gait_label,(num,1))
    total_slope_label = np.tile(slopelabel,(1,num))
    total_speed_label = np.tile(speedlabel,(1,num))
    for shift in shift_list:
        
        window_data2 = copy.deepcopy(window_data)
        window_data2[:,:,[0,2]] = window_data[:,:,[0,2]]+shift
        # window_data2[:,:,[1,3]] = window_data[:,:,[1,3]]/2
        total_window_data=np.concatenate((total_window_data,window_data2),axis=0)
    total_window_data=np.delete(total_window_data,0,0)
    
    return total_window_data,total_time_data,total_gait_label,total_slope_label,total_speed_label

# windowdata,timedata,angmin,angvelmin = data_scaler(window_data2,time_data2)
# windowdata2,timedata2,angmin2,angvelmin2 = data_scaler(window_data,time_data,angmin,angvelmin,test=True)

## function augment_RL
def augment_RL(data):
    window_data,time_data,gait_label, slopelabel, speedlabel = data
    window_data2 = copy.deepcopy(window_data)
    time_data2 = copy.deepcopy(time_data)
    gait_label2 = copy.deepcopy(gait_label)
    window_data2[:,:,[0,1]] = window_data[:,:,[2,3]] 
    window_data2[:,:,[2,3]] = window_data[:,:,[0,1]]
    time_data2[:,[0,1]] = time_data[:,[2,3]] 
    time_data2[:,[2,3]] = time_data[:,[0,1]]
    gait_label2[:,[0,1]] = gait_label[:,[2,3]] 
    gait_label2[:,[2,3]] = gait_label[:,[0,1]]
    window_data_total = np.concatenate((window_data,window_data2),axis=0)
    time_data_total = np.concatenate((time_data,time_data2),axis =0)
    gait_label_total = np.concatenate((gait_label,gait_label2),axis =0)
    slopelabel_total = np.concatenate((slopelabel,slopelabel),axis =0)
    speedlabel_total = np.concatenate((speedlabel,speedlabel),axis =0)
    return window_data_total,time_data_total,gait_label_total,slopelabel_total,speedlabel_total

def butter_bandpass(lowcut,highcut,fs,order=4):
    nyq= 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a= butter(order,[low,high],btype='band')
    return b,a
def butter_bandpass_filter(data, lowcut, highcut,fs,order=4):
    b,a = butter_bandpass(lowcut,highcut,fs,order=order)
    y=lfilter(b,a,data)
    return y
def butter_highpass(cutoff,fs,order=4):
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b,a = butter(order,normal_cutoff,btype='high',analog=False)
    return b,a
def butter_highpass_filter(data, highcut,fs,order=4):
    b,a = butter_highpass(highcut,fs,order=order)
    y=filtfilt(b,a,data)
    return y
#%%

def data_highpass(data,highcut,fs,order=4): 
    filtered_window = copy.deepcopy(data)
    for i in range( np.shape(data)[0]):
        filtered_window[i,:,0] = butter_highpass_filter(data[i,:,0],highcut,fs,order=4)
        filtered_window[i,:,1] = butter_highpass_filter(data[i,:,1],highcut,fs,order=4)
        filtered_window[i,:,2] = butter_highpass_filter(data[i,:,2],highcut,fs,order=4)
        filtered_window[i,:,3] = butter_highpass_filter(data[i,:,3],highcut,fs,order=4)
    return filtered_window

# import matplotlib.pyplot as plt


# data =make_dataset(window_size=200,
#                                         stride=1, trial_list=['1'] , Inclination_list=['02','10'] , WS_list=['2'] ,scaledata=False, train_true =False)
# data_shift = augment_shift(data,[1,2])

# test_data_filtered=data_highpass(test_data,0.05,100,order=4)
# test_time_data_filter =butter_highpass_filter(test_time_data[:,0], 0.05,100,order=4)
# plt.figure()
# plt.plot(test_data_filtered[399,:,0])
# plt.ylim([-50,50])
# plt.figure()
# plt.plot(test_time_data_filter[200:399])
# plt.ylim([-50,50])

# plt.figure()
# plt.plot(test_data[399,:,0])

# plt.figure()
# plt.plot(test_time_data[200:399,0])
# num =np.shape(test_data)[0]
# for i in range(2):
#     sliding_window= test_data[i]
#     filtered_swR= butter_highpass_filter(sliding_window[:,0],0.05,100,order=5)
#     filtered_swL= butter_highpass_filter(sliding_window[:,2],0.05,100,order=5)

#     # plt.plot(sliding_window[:,0])
#     plt.plot(filtered_swR)
#     plt.plot(filtered_swL)

#     plt.ylim([-60,40])




        
        
        
        
        
        