#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 10:25:16 2021

@author: Lobel001
"""

'''AFTER REVIEWERS COMMENTS: Figure 1 in JGR:Oceans manuscript is now only Ts for the 3 densities: Global sinking timescales (Ts) of sizes 1 mm to 0.1 um in 2004 (season averages)'''
'''SI plots have been merged into Fig. 1: diff initial plastic densities (30 and 840 kgm-3)'''

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np 
import cartopy
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
from numpy import *
import os 
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings("ignore", "Mean of empty slice")

res = '2x2' # [deg]: resolution of the global release of particles
loc = 'global'

dirwritefigs = '/Users/Lobel001/Desktop/Local_postpro/Kooi_figures/post_pro/'
dirwrite = '/Users/Lobel001/Desktop/Local_postpro/Kooi_data/post_pro_data/'

#%%

yr = 2004
numyrs = 1
seas = ['DJF', 'MAM', 'JJA', 'SON']
num_part = 9620

''' prepare cartopy projection and gridspec'''

projection = cartopy.crs.PlateCarree(central_longitude=72+180)
plt.rc('font', size = 16)

sizes = np.arange(3,8)
rhos = ['30','840','920']
fig_w = 14 #10 
fig_h = 13 #14 
fig = plt.figure(figsize=(fig_w,fig_h), constrained_layout=True)
gs = fig.add_gridspec(figure = fig, nrows = 6, ncols = 3, height_ratios=[6,6,6,6,6,1]) 
    
for col in range(3):  
    for row in range(5): 
        i = row*3 + col
        rho = rhos[col]
        si = sizes[row] 
        size = 'r1e-0'+str(si)     
       
        t_set_all = np.zeros((num_part,len(seas))) 
        vs_max_all = np.zeros((num_part,len(seas)))
        
        for idy,s in enumerate(seas):  
            print(f'rho = {rho}', size, s, yr)
            
            fname = 'global_%s_%s_3D_grid2x2_allrho_allr_90days_30dtsecs_12hrsoutdt.nc' % (s, yr)
            dirread = '/Users/Lobel001/Desktop/Local_postpro/Kooi_data/data_output/allrho/res_'+res+'/allr/'
            if not os.path.isfile(dirread+fname):
                print('%s not found' %fname)
            else:          
                data = Dataset(dirread+fname,'r')  
                time = data.variables['time'][0,:]/86400
                rho2 = float(rho) 
                size2 = float(size[1:len(size)]) 
                rho_output=data.variables['rho_pl'][:]
                r_output=data.variables['r_pl'][:]
               
                inds = np.where((rho_output==rho2) & (r_output.data==size2))[0]
               
                lons=data.variables['lon'][inds]
                lats=data.variables['lat'][inds] 
                depths =data.variables['z'][inds]
                vs = data.variables['vs'][inds]
                vs_init= data.variables['vs_init'][inds]
                w = data.variables['w'][inds]
            
            time = time - time[0]
    
            ''' Defining Ts using depth Vs + w > 0 m/s (particle's velocity + vertical advection is downward)'  '''
            vs_init2 = []
            w2 = []
            w_vs = []
            w_ind = []
            z_set2 = []
            t_set = np.zeros(depths.shape[0]) 
            t_set[:] = np.nan 
            boo = np.zeros((depths.shape[0],depths.shape[1]))
            
            for i in range(depths.shape[0]): #9620: number of particles 
                vs_init2 = vs_init[i,:]
                w2 = w[i,:]
                w_vs = vs_init2 + w2 
                w_ind = np.where(w_vs>0)
                
                z_set2 = []
                if w_ind[0].any():
                    z_set2 = w_ind[0][0]-1   
                    
                    t_set[i] = time[z_set2]
                    
            t_set = np.array(t_set)  
            t_set_all[:,idy] = t_set   
    
        Ts = np.nanmean(t_set_all,axis=1) 
        Ts[isnan(Ts)] = 100.   
    #%%    
        ''' plot: first columm is Ts'''
        cmap1 = plt.cm.get_cmap('magma_r', 9)
        c = col
        r = int(si)-3
        letter = chr(ord('a') + r*3 + c)
    
        ax = fig.add_subplot(gs[r,c], projection=projection)
        ax.coastlines(resolution='50m',zorder=3)
        ax.add_feature(cartopy.feature.LAND, color='lightgrey', zorder=2)
        ax.set_ylim([-70, 80])
        
        if r == 0:
            size_name = '1 mm'
            Ts03 = Ts
        if r == 1:
            size_name = '0.1 mm'
            Ts04 = Ts
        if r == 2:
            size_name = '10 \u03bcm'
            Ts05 = Ts
        if r == 3:
            size_name = '1 \u03bcm'
            Ts06 = Ts
        if r == 4 :
            size_name = '0.1 \u03bcm'
            Ts07 = Ts


            
        if col ==0:
            ax.text(-0.20, 0.4, f'radius = \n{size_name}', va='bottom', ha='center',
                rotation='horizontal', rotation_mode='anchor',
                transform=ax.transAxes)
        scat = ax.scatter(lons[:,0], lats[:,0], marker='.', c=Ts,cmap = cmap1, vmin = 0, vmax = 90, s = 10,zorder=1,transform = cartopy.crs.PlateCarree()) #scat = 
        if r == 0: 
            ax.title.set_text(rf'$\rho$ = {rho} kg m$^-$$^3$'+f'\n({letter})') #' radius = '+size_name) #$T_s$ 
        else:
            ax.title.set_text(f'({letter})')   
        
        if row == 4:
            cbaxes = fig.add_axes([0.15, 0.03, 0.75, 0.01])
            cbar = plt.colorbar(scat, cax=cbaxes, orientation="horizontal", aspect=50, extend='max')
            cbar.set_label(label='[days]') #, size=18)
        plt.tight_layout()
        
        '''UNCOMMENT THIS TO GET EXACT VALUES FOR MEDIAN (and std) TS TO ADD TO MANUSCRIPT '''
        # Ts[Ts==100] = np.nan
        # print(f'Median Ts for global {size_name} particles, density: {rho} = '+str(np.nanmedian(Ts.ravel().data))+': +/-'+str(np.nanstd(Ts.ravel().data)))

#%%

'''' Uncomment to save variables - NB now I've converted NaN (no sinking within 90 days) to 100 days.'''       
# with open(dirwrite+'90day_global_Ts_for03and07.pickle', 'wb') as f:
#     pickle.dump([lons,lats,Ts03,Ts07], f)

# with open(dirwrite+'90day_rho'+rho+'_global_Ts_allsizes.pickle', 'wb') as f: # 30th Nov, now saving for all sizes to be used for 3-yr sims 
#     pickle.dump([lons,lats,Ts03,Ts04,Ts05,Ts06,Ts07], f)

# with open(dirwrite+'Ts2004_allsizes_for_diffplot_with_advection.pickle', 'wb') as f:
#     pickle.dump([lons,lats,Ts02,Ts03,Ts04,Ts05,Ts06,Ts07], f)

'''Save the figure from the Spyder plot panel, top-right'''
