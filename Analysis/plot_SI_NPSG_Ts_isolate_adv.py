#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 11:01:35 2020

@author: Lobel001
"""

''' For JGR: Oceans SI plots: to check the separate effects of horizontal and vertical advection in NPSG'''
#Uncomment some sections to see the vertical velocity and attached algal growth
    # RESULTS of attached algae: When particles stay at rim of gyre (no horizontal advection), a lot more algae is present, so more of the larger particles sink. 

import matplotlib.pyplot as plt
from netCDF4 import Dataset
import numpy as np 
import cartopy
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import os 
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import pickle 
np.seterr(divide='ignore', invalid='ignore')
import warnings
import pandas as pd
from numpy import *
warnings.filterwarnings("ignore", "Mean of empty slice")

rho =  '920' # [kgm-3]: density of the plastic 
adv_type = 'Horizontal advection only' # '3D advection' 'Vertical advection only' 'Horizontal advection only'

if adv_type == '3D advection':
    fname = '3yr_NPac_3D_grid2x2_allrho_allr_1000days_60dtsecs_12hrsoutdt.nc'
elif adv_type == 'Vertical advection only':
    fname = '3yr_wAdvOnly_NPac_3D_grid2x2_allrho_allr_1000days_60dtsecs_12hrsoutdt.nc'    
elif adv_type == 'Horizontal advection only':
    fname = '3yr_uvAdvOnly_NPac_3D_grid2x2_allrho_allr_1000days_60dtsecs_12hrsoutdt.nc'  

dirread = '/Users/Lobel001/Desktop/Local_postpro/Kooi_data/data_output/allrho/res_2x2/allr/'
dirread_Ts = '/Users/Lobel001/Desktop/Local_postpro/Kooi_data/post_pro_data/'
dirwritefigs = '/Users/Lobel001/Desktop/Local_postpro/Kooi_figures/post_pro/'

latmin = 28
latmax = 36
lonmin = -143
lonmax = -135

''' Ts for 90-day sims to have background patch of STG'''
with open(dirread_Ts+'90day_rho'+rho+'_global_Ts_allsizes.pickle', 'rb') as f:
    lons_global,lats_global,Ts03,Ts04,Ts05,Ts06,Ts07 = pickle.load(f)

''' prepare cartopy projection and gridspec'''
cmap0 = plt.cm.get_cmap('magma_r', 10)
cmap = plt.cm.get_cmap('magma_r', 10)#cmo.haline

projection = cartopy.crs.PlateCarree(central_longitude=180) #central_longitude=72+180)
plt.rc('font', size = 20)
sizes = np.arange(3,8)
fig_w = 20 
fig_h = 12 
fig = plt.figure(figsize=(fig_w,fig_h)) #, constrained_layout=True) # 10,5
gs = fig.add_gridspec(figure = fig, nrows = 3, ncols = 3, height_ratios=[7,7,1]) #,hspace = 0.5)

'''Uncomment to see the vertical velocity and attached algal growth'''
# fig0 = plt.figure(figsize=(fig_w,fig_h)) 
# gs0 = fig0.add_gridspec(figure = fig0, nrows = 3, ncols = 3, height_ratios=[7,7,1])

# fig1 = plt.figure(figsize=(fig_w,fig_h)) 
# gs1 = fig1.add_gridspec(figure = fig1, nrows = 3, ncols = 3, height_ratios=[7,7,1])

# fig2 = plt.figure(figsize=(fig_w,fig_h)) 
# gs2 = fig2.add_gridspec(figure = fig2, nrows = 3, ncols = 3, height_ratios=[7,7,1])

for row in range(2): 
    for col in range(3): 
        ii = row*3 + col
        
        if ii <5:
        
            si = sizes[ii] #row]
            size = 'r1e-0'+str(si) 
            Ts_90d = eval(f'Ts0{si}')
            Ts_90d[Ts_90d<100]= np.nan
            print(size)
        
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
               
                time_all = (data.variables['time'][inds,:]/86400)        
                lons=data.variables['lon'][inds]#+72+180
                lats=data.variables['lat'][inds] 
                depths =data.variables['z'][inds]
                vs = data.variables['vs'][inds]
                vs_init= data.variables['vs_init'][inds]
                w = data.variables['w'][inds]
                a = data.variables['a'][inds]
        
                vs_init2 = []
                w2 = []
                w_vs = []
                w_ind = []
                z_set = []
                Ts = np.zeros(depths.shape[0]) 
                Ts[:] = np.nan 
                lon_sink = np.zeros(depths.shape[0])
                lat_sink = np.zeros(depths.shape[0])
                lons_tosink = np.zeros((depths.shape[0],depths.shape[1]))
                lats_tosink = np.zeros((depths.shape[0],depths.shape[1]))
                lons_tosink[:,:] = np.nan
                lats_tosink[:,:] = np.nan
                
                for i in range(depths.shape[0]): #9620: number of particles 
                    time = time - time[0]
                    
                    '''Uncomment to check the vertical velocity and attached algal growth - showing 3 peaks in MAM '''
                    # date = pd.to_datetime("1st of Jan, 2004") 
                    # date_series = date + pd.to_timedelta(time, 'D')
                    # date_form = DateFormatter("%m/%y")
                    
                    # plt.rc('font', size = 12)
                    # ax0 = fig0.add_subplot(gs0[row,col])
                    # ax0.plot(date_series,w[i,:]) 
                    # #ax0.set_xticks(date_series,date_series, rotation ='vertical')
                    # ax0.xaxis.set_major_formatter(date_form)
                    
                    # ax1 = fig1.add_subplot(gs1[row,col])
                    # ax1.plot(date_series,vs[i,:]) 
                    # #ax1.set_xticks(date_series,date_series, rotation ='vertical')
                    # ax1.xaxis.set_major_formatter(date_form)
                    
                    # ax2 = fig2.add_subplot(gs2[row,col])
                    # ax2.plot(date_series,a[i,:]) 
                    # #ax2.set_xticks(date_series,date_series, rotation ='vertical')
                    # ax2.xaxis.set_major_formatter(date_form) 
                    
                    if adv_type == 'Horizontal advection only': #since Ts should exclude vertical advection (w) for uv adv only 
                        vs2 = vs[i,:]
                        w_ind = np.where(vs2>0)
                    else:
                        vs_init2 = vs_init[i,:]
                        w2 = w[i,:]
                        w_vs = vs_init2 + w2 
                        w_ind = np.where(w_vs>0)
  
                    z_set2 = []
                    if w_ind[0].any():
                        z_set = w_ind[0][0]-1   
                        
                        Ts[i] = time[z_set]
                        lon_sink[i] = lons[i,z_set]
                        lat_sink[i] = lats[i,z_set]
                        
                        lons_tosink[i,:] = lons[i,:]
                        lats_tosink[i,:] = lats[i,:]
                        
                        lons_tosink[i,z_set+1:] = np.nan
                        lats_tosink[i,z_set+1:] = np.nan
                        
                        
                Ts[isnan(Ts)] = 1500.   
        
                if si == 3:
                    size_name = '1 mm'
                if si == 4:
                    size_name = '0.1 mm'
                if si == 5:
                    size_name = '10 \u03bcm'
                if si == 6:
                    size_name = '1 \u03bcm'
                if si == 7:
                    size_name = '0.1 \u03bcm'
            plt.rc('font', size = 20)
            letter = (chr(ord('a') + ii))
            ax = fig.add_subplot(gs[row,col], projection=projection)
            ax.coastlines(resolution='50m',zorder=3)
            ax.add_feature(cartopy.feature.LAND, color='lightgrey', zorder=2)
            ax.set_extent([lonmax+10,lonmin-10,latmin-6,latmax+6]) #-155,-125,22,40]) #[179, -109, 10, 40])
            
            ax.scatter(lons_global[:,0], lats_global[:,0], marker='.', c=Ts_90d,cmap = cmap0, alpha = 0.5, vmin = 0, vmax = 90, s = 1900,zorder=0,transform = cartopy.crs.PlateCarree()) 
            scat = ax.scatter(lons_tosink,lats_tosink, c = time_all, edgecolors = 'k', vmin = 0, vmax = 1000,  marker='.', s = 4, zorder = 2, transform = cartopy.crs.PlateCarree(),cmap = cmap) 
            rect= plt.Rectangle((lonmin-2,latmin-2), 12, 12, ec = 'black', lw = 6, fc = 'none', zorder = 3, transform = cartopy.crs.PlateCarree())
            ax.add_patch(rect)
            ax.scatter(lon_sink, lat_sink, marker= '.', c = Ts, edgecolors = 'k', cmap = cmap, vmin = 0, vmax = 1000, s = 800,zorder=4, transform = cartopy.crs.PlateCarree()) #"$\u25A1$"
            ax.title.set_text(f'({letter}) radius = {size_name}')
        
            gl = ax.gridlines(crs=cartopy.crs.PlateCarree(), draw_labels=False, linewidth=0.5,
                                  color='gray', alpha=0.5, linestyle='--')
            gl.xlocator = mticker.FixedLocator(np.arange(lonmin-26,lonmax+26,12)) #-185, -115, 10))
            gl.ylocator = mticker.FixedLocator(np.arange(latmin-26,latmax+26,12)) #(15, 65, 10))
            gl.xlabels_bottom = True
            gl.xformatter = LONGITUDE_FORMATTER
            gl.ylabels_left = True
            gl.yformatter = LATITUDE_FORMATTER
        else:
            cbaxes = fig.add_axes([0.1, 0.15, 0.8, 0.01])
            cbar = plt.colorbar(scat, cax=cbaxes, orientation="horizontal", aspect=50, extend='neither')
            cbar.set_label(label='[days]')

fig.suptitle(adv_type)

'''Uncomment to check the vertical velocity and attached algal growth - showing 3 peaks in MAM '''
# fig0.suptitle('NEMO w velocity')
# fig1.suptitle('Vs settling velocity')
# fig2.suptitle('Algal conc')