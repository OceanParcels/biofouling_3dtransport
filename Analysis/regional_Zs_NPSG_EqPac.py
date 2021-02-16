#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 20:11:04 2021

@author: Lobel001
"""

'''For resubmission: decided to add Zs for Fig. 6 now in JGR:Oceans manuscript for regional analyses (North Pacific Subtropical Gyre and Equatorial Pacific)'''

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np 
import cartopy
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from numpy import *
import os 
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.filterwarnings("ignore", "Mean of empty slice")

'''CHOOSE intial density and size, but for manuscript, only rho = 920 and s = 6'''
rho =  '920' # [kgm-3]: density of the plastic #920 #840 #30
s = '6' # for 10-6 m 

size = f'r1e-0{s}'
size_ttl = f"1e-{s} m"

if s == '3':
   size_ttl = '1 mm'
if s == '4':
    size_ttl = '0.1 mm'
if s == '5':
    size_ttl = '10 \u03bcm'
if s == '6':
    size_ttl = '1 \u03bcm'
if s == '7':
    size_ttl = '0.1 \u03bcm'


dirread = '/Users/Lobel001/Desktop/Local_postpro/Kooi_data/data_output/allrho/res_2x2/allr/'
dirwritefigs = '/Users/Lobel001/Desktop/Local_postpro/Kooi_figures/post_pro/'
dirwrite = '/Users/Lobel001/Desktop/Local_postpro/Kooi_data/post_pro_data/'


#%%

seas = ['MAM'] #'DJF', 'MAM', 'JJA', 'SON']
proc = ['bfadv','nobf','noadv'] #bfadv', 
region = ['NPSG','EqPac'] 

num_part = 25

''' prepare cartopy projection and gridspec'''

projection = cartopy.crs.PlateCarree(central_longitude=72+180)
plt.rc('font', size = 24)

fig_w = 20
fig_h = 15
fig = plt.figure(figsize=(fig_w,fig_h), constrained_layout=True) # 10,5
gs = fig.add_gridspec(figure = fig, nrows = 3, ncols = 3, height_ratios=[4,4,1]) #,hspace = 0.5, wspace = 0.5) #, height_ratios=[10,1,10,1], wspace = 0.05, hspace = 1) # gridspec.GridSpec
    
  
for row in range(2): 
    for col in range(3):
        ind = row*3 + col
        p = proc[col] #process 
        r = region[row]
       
        t_set_all = np.zeros((num_part,len(seas))) 
        t_set_all[:] = np.nan
        
        if r == 'NPSG':
            latmin = 26
            latmax = 38
            lonmin = -133
            lonmax = -145
        elif r == 'EqPac':
            latmin = -6
            latmax = 6
            lonmin = -138
            lonmax = -150
            
        for idy,s in enumerate(seas):
        
            print(size, s, p, r)

            fname = f'regional_{r}_{p}_{s}_2004_3D_grid2x2_allrho_allr_90days_30dtsecs_12hrsoutdt.nc'
            
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
                    
                    
                    ''' Defining Zs '''
                    z2 = [0,] * depths.shape[1]
                    for ii in range(z_set2,depths.shape[1]): 
                        z2[ii-1] = depths[i,ii]-depths[i,ii-1] 
                    zf = np.where(np.array(z2) < 0.)[:]           
                    zind = zf[0][0] if np.array(zf).any() else []
                    if np.array(zind).any():
                        j = np.array(zind)
                        boo[i,:j+1] = 1.

            plot_idx = np.random.permutation(depths.shape[0])
            
            time_p = np.tile(time,(depths.shape[0],1))
            lats_p = np.tile(lats[plot_idx,0],(lats.shape[1],1)).T
            depths_p = depths[plot_idx,:]
            boo2 = np.array(boo[plot_idx,:],dtype='bool')


            ''' using colorbar from above to separate colours by initial release latitudinal bins '''
            
            ax = fig.add_subplot(gs[row,col])
            
            time_save = np.zeros(plot_idx.shape[0])
            time_save[:] = np.nan
            depths_save = np.zeros(plot_idx.shape[0])
            depths_save[:] = np.nan  
            lats_save = np.zeros(plot_idx.shape[0])
            d = np.zeros((depths.shape[0],depths.shape[1]))
            d[:] = np.nan 
            for ii in range(plot_idx.shape[0]):
                boo_p = boo2[ii,:]
          
                """To plot a scatter dot for Zs"""
                    
                ax.plot(time_p[ii,boo_p],(depths_p[ii,boo_p]*-1), c = 'grey', linewidth=3, alpha = 0.7)
                ind_nonan = np.where(depths_p[ii,boo_p]>0.6)
                if np.array(ind_nonan).any():
                    last_ind = ind_nonan[0][-1]
                    ax.plot(time_p[ii,last_ind],(depths_p[ii,last_ind]*-1), marker = 'o', c = 'grey', markersize=15, linewidth = 1, markeredgecolor='black',alpha = 0.7, markeredgewidth=3) 
                
                    time_save[ii] = time_p[ii,last_ind]
                    depths_save[ii] = depths_p[ii,last_ind]
                    lats_save[ii] = lats_p[ii,0]
            
            if row == 0:
                ax.set_ylim(top=-0.5, bottom =-2) 
            else:
                ax.set_ylim(top=2, bottom =-90) 
            ax.set_xlim(left=0, right = 90)

        plt.rc('font') 

        if row == 1 and col ==1:
            ax.set_xlabel('Time [days]')
            
        if col == 0:
            ax.set_ylabel('Depth [m]')
            proc_ttl = 'original'
        if col == 1:
            proc_ttl = 'no biofouling'
        if col == 2:
            proc_ttl = 'no advection'
        
        letter = (chr(ord('a') + row*3+col))
        
            
        ax.title.set_text(f'({letter}) {r}: {proc_ttl}')

plt.suptitle(f'radius = {size_ttl} in {s} \n', fontsize=34)