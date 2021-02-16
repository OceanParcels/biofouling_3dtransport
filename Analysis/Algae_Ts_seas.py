#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 23:38:26 2020

@author: Lobel001
"""
'''For the resubmision, this became Fig. 3 in JGR:Oceans draft: MEDUSA surface algal concentrations and Ts for the 4 seasons'''

import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy
import cmocean.cm as cmo
import os
from numpy import *

# Added this (from a forum) as a fix to error I was getting regarding 'GeoAxes not having a _hold function'
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

'''CHOOSE SIZE TO PLOT Ts (in paper it's now 1e-6 for resubmimssion: and 1e-4m is in SI)'''
size = 'r1e-06'
rho = '920'

dirread = '/Users/Lobel001/Desktop/Local_postpro/Kooi_data/NEMO_phys_params/'
dirread_Ts = '/Users/Lobel001/Desktop/Local_postpro/Kooi_data/data_output/allrho/res_2x2/allr/'

''' Preparing the projection and subplots'''
plt.rc('font', size = 14)
projection = cartopy.crs.PlateCarree(central_longitude=72+180) 
fig_w = 10 
fig_h = 11 
fig = plt.figure(figsize=(fig_w,fig_h)) 
gs = fig.add_gridspec(figure = fig, nrows = 5, ncols = 2, height_ratios=[6,6,6,6,1])

seasnames = ['DJF', 'MAM', 'JJA', 'SON']
monvals = ['12','01','02','03','04','05','06','07','08','09','10','11']

for row in range(4): 
    for col in range(2):
        i = row*2 + col
        seas = seasnames[row]
        
        print(seas)

        yr0 = '2004'
        yr1 = '2004'

        ai = fig.add_subplot(gs[row, col], projection=projection)
        ai.coastlines(resolution='50m',zorder=3)
        ai.add_feature(cartopy.feature.LAND, color='lightgrey', zorder=2)
        ai.set_ylim([-70, 80]) 

        if row == 0: 
            yr0 = '2003'
            mons = monvals[0:3]
        if row == 1: 
            mons = monvals[3:6]
        if row == 2: 
            mons = monvals[6:9]
        if row == 3: 
            mons = monvals[9:12]

        if i ==0 or i == 2 or i ==4 or i ==6:
 
            fname = dirread+'algal_conc_2004_'+seas+'.nc'
            M = xr.open_dataset(fname).__xarray_dataarray_variable__  
            M = M.assign_coords(nav_lat=M.nav_lat)
            M = M.assign_coords(nav_lon=M.nav_lon) 

            a = M.plot(ax=ai, x='nav_lon', y='nav_lat', add_colorbar=False, vmin=0, vmax=4e7, rasterized=True, cmap=cmo.algae, zorder = 1, transform=cartopy.crs.PlateCarree()) 
            title = 'Algal conc. in %s' % seas

        
        if i ==1 or i == 3 or i ==5 or i ==7:
            cmap = plt.cm.get_cmap('magma_r', 9) 
            fname = 'global_'+seas+'_2004_3D_grid2x2_allrho_allr_90days_30dtsecs_12hrsoutdt.nc' 
            
            if not os.path.isfile(dirread_Ts+fname):
                print('%s not found' %fname)
            else:          
                data = Dataset(dirread_Ts+fname,'r')  
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
            
            t_set[isnan(t_set)] = 100.
         
            a2 = ai.scatter(lons[:,0], lats[:,0], marker='.', c=t_set,cmap = cmap, vmin = 0, vmax = 90, s = 10, zorder=1,transform = cartopy.crs.PlateCarree()) #,crs=cartopy.crs.PlateCarree()) #scat = 

            title = '$T_s$ in %s' % seas
            

        ai.set_title('%s) %s ' % (chr(ord('a') + row*2 + col), title))

cbaxes = fig.add_axes([0.05, 0.045, 0.4, 0.015]) # defines the x, y, w, h of the colorbar 
plt.colorbar(a, cax=cbaxes, orientation="horizontal", aspect=100, extend='max', label='[no. m$^{-3}$]', use_gridspec=True)

cbaxes2 = fig.add_axes([0.55, 0.045, 0.4, 0.015]) 
plt.colorbar(a2, cax=cbaxes2, orientation="horizontal", aspect=100, extend='max', label='[days]', use_gridspec=True) 

fig.canvas.draw()
plt.tight_layout()
'''No longer saving figures this way, simply saving from the Spyder plot panel, top-right'''
#plt.savefig('/home/dlobelle/Kooi_figures/non_parcels_output/NEMO_MLD_algae_allseas_2004.pdf')