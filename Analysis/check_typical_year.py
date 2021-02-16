#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 09:33:12 2020

@author: Lobel001
"""

'''finding the 'most typical year' '''
#input here is from season_av_global_Ts_Zm_v2.py

import matplotlib.pyplot as plt
import numpy as np 
import pickle 
np.seterr(divide='ignore', invalid='ignore')
import warnings
import numpy.matlib
warnings.filterwarnings("ignore", "Mean of empty slice")

dirwrite = '/Users/Lobel001/Desktop/Local_postpro/Kooi_data/post_pro_data/'

    
with open(dirwrite+'920r1e-07allseas_Ts_sum.pickle','rb') as f: 
    Ts_sum7 = np.array(pickle.load(f))[0,:]
with open(dirwrite+'920r1e-05allseas_Ts_sum.pickle','rb') as f: 
    Ts_sum5 = np.array(pickle.load(f))[0,:]    
with open(dirwrite+'920r1e-04allseas_Ts_sum_test.pickle','rb') as f: 
    Ts_sum4 = np.array(pickle.load(f))[0,:]

norm4 = (Ts_sum4-np.mean(Ts_sum4))/np.std(Ts_sum4)
norm5 = (Ts_sum5-np.mean(Ts_sum5))/np.std(Ts_sum5)
norm7 = (Ts_sum7-np.mean(Ts_sum7))/np.std(Ts_sum7)

yrs = np.arange(2001,2011)

plt.plot(yrs,norm4)
plt.plot(yrs,norm5)
plt.plot(yrs,norm7)

a = np.vstack((norm4,norm5,norm7))

tot_sum = np.nansum(a, axis=0)
bestyr = yrs[np.argmin(tot_sum)]

print(f'most typical year = {bestyr}')