import numpy as np
import math


#%%
def Gait_label(HSL,HSR):     # input: data size, [HSR HSL]    output: R[x y] L[x y]

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

    if HSR[0]!=0:
        phasedataR = np.zeros((int(HSR[0])-1))#앞의 모든것은 0
    else:
        phasedataR = np.array([])
        
    for i in range(len(stridetimeR)):
        pdata = np.linspace(0,1,int(stridetimeR[i]+1)) # 0 ~ 1

        pdata = np.delete(pdata, len(pdata)-1,0) # 1삭제
        phasedataR = np.append(phasedataR,pdata) # 추
        
    phasedataR = np.append(phasedataR ,1) # 마지막 1 추가
    phasedataR = phasedataR[:n] # 마지막 Index까지
    theta = phasedataR*2*math.pi
    x = np.cos(theta)
    y = np.sin(theta)
    true_phase[2,:] = x
    true_phase[3,:] = y
    
    #반복
    if HSL[0]!=0:
        phasedataL =np.zeros((int(HSL[0])-1))#앞의 모든것은 0
    else:
        phasedataL = np.array([])
        
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
    true_phase[0,:] = x
    true_phase[1,:] = y
    true_phase = np.transpose(true_phase)
    
    
    
    return first_index,true_phase
