#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon Nov 16 23:38:26 2020

@author: Lobel001
"""
'''For the resubmision, this became Fig. 4 (instead of Figure 3) in JGR:Oceans draft: MEDUSA surface algal concentrations and Ts for the 4 seasons'''

import numpy as np
import xarray as xr
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import cartopy
import cmocean.cm as cmo
import os

# Added this (from a forum) as a temporary fix to error I was getting regarding 'GeoAxes not having a _hold function'
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

#CHOOSE SIZE TO PLOT Ts (in paper it's now 1e-6 for resubmimssion: and 1e-4m is in SI)
size = 'r1e-06'
rho = '920'

dirread = '/Users/Lobel001/Desktop/Local_postpro/Kooi_data/NEMO_phys_params/'
dirread_Ts = '/Users/Lobel001/Desktop/Local_postpro/Kooi_data/data_output/allrho/res_2x2/allr/'

''' Preparing the projection and subplots'''

plt.rc('font', size = 14)
projection = cartopy.crs.PlateCarree(central_longitude=72+180)
fig_w = 10 
fig_h = 11 
fig = plt.figure(figsize=(fig_w,fig_h), constrained_layout=True) 
gs = fig.add_gridspec(figure = fig, nrows = 5, ncols = 2, height_ratios=[4,4,4,4,1]) 

seasnames = ['DJF', 'MAM', 'JJA', 'SON']
monvals = ['12','01','02','03','04','05','06','07','08','09','10','11']
    
    
for row in range(4): 
    for col in range(2): 
        i = row*2 + col
        seas = seasnames[row]
        
        print(seas)

        yr0 = '2004'
        yr1 = '2004'
        
        if col ==0:
            ai = fig.add_subplot(gs[row, col], projection=projection)
            ai.coastlines(resolution='50m',zorder=3)
            ai.add_feature(cartopy.feature.LAND, color='lightgrey', zorder=2)
            ai.set_ylim([-70, 80])            
        else:
            ai = fig.add_subplot(gs[row,col])

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
            fname = dirread+'MLD_2004_'+seas+'.nc'
            M0 = xr.open_dataset(fname).mldr10_1  
            M0 = M0.assign_coords(nav_lat=M0.nav_lat)
            M0 = M0.assign_coords(nav_lon=M0.nav_lon) 

            a2 = M0.plot(ax=ai, x='nav_lon', y='nav_lat', add_colorbar=False, vmin=0, vmax=200, rasterized=True, cmap=cmo.deep, zorder = 1, transform=cartopy.crs.PlateCarree()) 
            title = 'MLD in %s' % seas

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
                    
                    ''' Defining Zs '''
                    
                    z2 = [0,] * depths.shape[1]
                    for ii in range(z_set2,depths.shape[1]): 
                        z2[ii-1] = depths[i,ii]-depths[i,ii-1] 
                    zf = np.where(np.array(z2) < 0.)[:]           
                    zind = zf[0][0] if np.array(zf).any() else [] 
                    if np.array(zind).any():
                        j = np.array(zind)
                        boo[i,:j+1] = 1.
                        
            ''' to get the colormap for line plot below, need to get it from scatterplot (figures produced in plot pane, not sure how to prevent that)'''
            plot_idx = np.random.permutation(depths.shape[0])
            
            time_p = np.tile(time,(depths.shape[0],1))#.T
            lats_p = np.tile(lats[plot_idx,0],(lats.shape[1],1)).T
            depths_p = depths[plot_idx,:]
            boo2 = np.array(boo[plot_idx,:],dtype='bool')
        
            
            fig1 = plt.figure(figsize=(fig_w,fig_h)) #15,10
            cmap = plt.cm.get_cmap('coolwarm',7)
            a3 = plt.scatter(time_p[boo2],(depths_p[boo2]*-1), vmin = -70, vmax = 70, c = lats_p[boo2], cmap = cmap)
            plt.colorbar()
            plt.ylim(top=0, bottom =-240) 
            plt.xlim(left=0, right = 90)
            ax = plt.gca()
            plt.ylabel('Depth [m]')
            plt.xlabel('Time [days]')


            ''' using colorbar from above to separate colours by initial release latitudinal bins'''
            
            
            time_save = np.zeros(plot_idx.shape[0])
            time_save[:] = np.nan
            depths_save = np.zeros(plot_idx.shape[0])
            depths_save[:] = np.nan  
            lats_save = np.zeros(plot_idx.shape[0])
            d = np.zeros((depths.shape[0],depths.shape[1]))
            d[:] = np.nan 
            for ii in range(plot_idx.shape[0]):
                boo_p = boo2[ii,:]
                
                if lats_p[ii,0]<-50.:
                    rgb = cmap(0)[:3]
                elif lats_p[ii,0]<-30. and lats_p[ii,0]>=-50.:
                    rgb = cmap(1)[:3]
                elif lats_p[ii,0]<-10. and lats_p[ii,0]>=-30.:
                    rgb = cmap(2)[:3]
                elif lats_p[ii,0]<10. and lats_p[ii,0]>=-10.:
                    rgb = cmap(3)[:3]            
                elif lats_p[ii,0]<30. and lats_p[ii,0]>=10.:
                    rgb = cmap(4)[:3]
                elif lats_p[ii,0]<50. and lats_p[ii,0]>=30.:
                    rgb = cmap(5)[:3]
                elif lats_p[ii,0]>=50.:
                    rgb = cmap(6)[:3]
          
                """To plot a scatter dot for Zs"""
                    
                ai.plot(time_p[ii,boo_p],(depths_p[ii,boo_p]*-1), c = rgb, linewidth=1, alpha = 0.6)
                ind_nonan = np.where(depths_p[ii,boo_p]>0.6)
                if np.array(ind_nonan).any():
                    last_ind = ind_nonan[0][-1]
                    ai.plot(time_p[ii,last_ind],(depths_p[ii,last_ind]*-1), marker = 'o', c = rgb, markersize=5, linewidth = 0, markeredgecolor='black',alpha = 0.6) 
                
                    time_save[ii] = time_p[ii,last_ind]
                    depths_save[ii] = depths_p[ii,last_ind]
                    lats_save[ii] = lats_p[ii,0]

            ai.set_ylim(top=0, bottom =-200) 
            ai.set_xlim(left=0, right = 90) 


            if row == 3:
                ai.set_ylabel('Depth [m]')
                ai.set_xlabel('Time [days]') 


            title = '$Z_s$ in %s' % seas
            
        ai.set_title('%s) %s ' % (chr(ord('a') + row*2 + col), title))
cbaxes2 = fig.add_axes([0.03, 0.03, 0.4, 0.015]) # defines the x, y, w, h of the colorbar 
plt.colorbar(a2, cax=cbaxes2, orientation="horizontal", aspect=100, extend='max', label='[m]', use_gridspec=True) 

bounds = [-50,-30,-10,10,30,50]
cbaxes3 = fig.add_axes([0.58, 0.03, 0.4, 0.015]) # defines the x, y, w, h of the colorbar 
plt.colorbar(a3, cax=cbaxes3, orientation="horizontal", aspect=50, extend='both', ticks = bounds, label='[initial latitude]') 

'''No longer saving figures this way, simply saving from the Spyder plot panel, top-right'''
#plt.savefig('/home/dlobelle/Kooi_figures/non_parcels_output/NEMO_MLD_algae_allseas_2004.pdf')