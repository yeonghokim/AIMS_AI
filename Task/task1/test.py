# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 22:30:03 2022

@author: yeong
"""

import scipy.io
import numpy as np

mat_file_name = "C:/Users/yeong/Desktop/학부연구생/task/sEMG(9).mat"
mat_file = scipy.io.loadmat(mat_file_name)

emg= mat_file['emg']
label  = mat_file['label']
rep = mat_file['repetition']
flag=1
print(emg[flag:flag+3,0])
type(emg[flag:flag+3,0])