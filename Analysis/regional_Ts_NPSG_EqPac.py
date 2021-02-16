#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 18:32:58 2020

@author: Lobel001
"""

'''For resubmission: now Figure 5 in JGR:Oceans manuscript: Ts for regional analyses (North Pacific Subtropical Gyre and Equatorial Pacific)'''
''' Only size 1 micron used for manuscript (since required most explanations)'''

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

seas = ['DJF', 'MAM', 'JJA', 'SON']
proc = ['bfadv','nobf','noadv'] #bfadv', 
region = ['NPSG','EqPac'] 

num_part = 25

''' prepare cartopy projection and gridspec'''

projection = cartopy.crs.PlateCarree(central_longitude=72+180)
plt.rc('font', size = 24)

fig_w = 20
fig_h = 20 
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
                rho2 = float(rho) #rho_all[ii]
                size2 = float(size[1:len(size)]) #size_all[ii]
                rho_output=data.variables['rho_pl'][:]
                r_output=data.variables['r_pl'][:]
               
                inds = np.where((rho_output==rho2) & (r_output.data==size2))[0]
               
                lons=data.variables['lon'][inds]#+72+180
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
                if p == 'noadv':
                    w2 = 0
                else:
                    w2 = w[i,:]
                w_vs = vs_init2 + w2 
                w_ind = np.where(w_vs>0)
                
                z_set2 = []
                if w_ind[0].any():
                    z_set2 = w_ind[0][0]-1   
                    
                    t_set[i] = time[z_set2]
            t_set_all[:,idy] = t_set        
        Ts = np.nanmean(t_set_all,axis=1) 
        Ts[isnan(Ts)] = 100. 
                    
        ''' plot: first columm is Ts'''
        cmap = plt.cm.get_cmap('magma_r', 9)
    
        ax = fig.add_subplot(gs[row,col], projection=projection)
        ax.coastlines(resolution='50m',zorder=3)
        ax.add_feature(cartopy.feature.LAND, color='lightgrey', zorder=2)
        ax.set_extent([lonmin, lonmax, latmin, latmax])
        
        if col == 0:
            proc_ttl = 'original'
        if col == 1:
            proc_ttl = 'no biofouling'
        if col == 2:
            proc_ttl = 'no advection'
        
        letter = (chr(ord('a') + row*3+col))
        
              
        scat = ax.scatter(lons[:,0], lats[:,0], marker='.', edgecolors = 'k', c=Ts,cmap = cmap, vmin = 0, vmax = 90, s = 1500,zorder=1,transform = cartopy.crs.PlateCarree()) #scat = 

        ax.title.set_text(f'({letter}) {r}: {proc_ttl}')
        
        gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=False, linewidth=0.5,
                color='gray', alpha=0.5, linestyle='--')
        gl.xlocator = mticker.FixedLocator(np.arange(lonmax-6,lonmin+6,4))
        gl.ylocator = mticker.FixedLocator(np.arange(latmin-6, latmax+6,4))
        gl.xlabels_bottom = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.ylabels_left = True
        gl.yformatter = LATITUDE_FORMATTER


plt.suptitle(f'radius = {size_ttl}', y=0.95, fontsize=34)          
cbaxes = fig.add_axes([0.15, 0.2, 0.75, 0.01])
plt.colorbar(scat, cax=cbaxes, orientation="horizontal", aspect=50, extend='max', label = '[days]')
