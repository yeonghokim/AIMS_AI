
import numpy as np
import math

#%% XY를 페이즈로 변환 
def XY2Theta(estimationX, estimationY):
    theta = np.arctan2(estimationX, estimationY) #-pi ~pi
    sn = theta < 0
    theta = theta + (sn * 1)*math.pi*2#0 ~ 2*pi
    Est = theta*(1/(2*math.pi)) # 0 ~ 1
    return Est

def XY2Gait(estimation, true):

    phase_EstL = XY2Theta(estimation[:, 1], estimation[:, 0])
    phase_EstR = XY2Theta(estimation[:, 3], estimation[:, 2])
    
    phase_labelL = XY2Theta(true[:, 1], true[:, 0])
    phase_labelR = XY2Theta(true[:, 3], true[:, 2])

    return phase_EstL, phase_labelL, phase_EstR, phase_labelR

def NRMSE(p,y):
    RMSE = np.sqrt((np.mean((p-y)**2)))/(np.max(y) - np.min(y))
    RMSE = round(RMSE,4)
    return RMSE

