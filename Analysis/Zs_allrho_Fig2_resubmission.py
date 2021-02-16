#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 15:33:55 2021

@author: Lobel001
"""

'''AFTER REVIEWERS COMMENTS: Figure 2 in JGR:Oceans manuscript is now only Zs for the 3 densities of sizes 1 mm to 0.1 um in 2004 (season averages)'''
'''SI plots have been merged into Fig. 2: diff initial plastic densities (30 and 840 kgm-3)'''


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
seas = ['MAM'] #['DJF', 'MAM', 'JJA', 'SON']
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
                w2 = w[i,:]
                w_vs = vs_init2 + w2 
                w_ind = np.where(w_vs>0)
                
                z_set2 = []
                if w_ind[0].any():
                    z_set2 = w_ind[0][0]-1   
                    
                    t_set[i] = time[z_set2]
                    
                    
                    ''' Defining Zs '''
                    #if s == 'MAM':
                    z2 = [0,] * depths.shape[1]
                    for ii in range(z_set2,depths.shape[1]): 
                        z2[ii-1] = depths[i,ii]-depths[i,ii-1] 
                    zf = np.where(np.array(z2) < 0.)[:]           
                    zind = zf[0][0] if np.array(zf).any() else []
                    if np.array(zind).any():
                        j = np.array(zind)
                        boo[i,:j+1] = 1.
                                
    #%%
            ''' plot: 2nd column is Zs for MAM only'''   
            #if s == 'MAM':
            c = col
            r = int(si)-3
            letter = chr(ord('a') + r*3 + c)

            if si == 3:
                size_name = '1 mm'
                #letter = 'b'
            if si == 4:
                size_name = '0.1 mm'
                #letter = 'd'
            if si == 5:
                size_name = '10 \u03bcm'
                #letter = 'f'
            if si == 6:
                size_name = '1 \u03bcm'
                #letter = 'h'
            if si == 7:
                size_name = '0.1 \u03bcm'
                #letter = 'j'
            
            ''' to get the colormap for line plot below, need to get it from scatterplot (figures produced in plot pane, not sure how to prevent that)'''
            plot_idx = np.random.permutation(depths.shape[0])
            
            time_p = np.tile(time,(depths.shape[0],1))
            lats_p = np.tile(lats[plot_idx,0],(lats.shape[1],1)).T
            depths_p = depths[plot_idx,:]
            boo2 = np.array(boo[plot_idx,:],dtype='bool')
            
            fig1 = plt.figure(figsize=(15,10))
            cmap = plt.cm.get_cmap('coolwarm',7)
            scat = plt.scatter(time_p[boo2],(depths_p[boo2]*-1), vmin = -70, vmax = 70, c = lats_p[boo2], cmap = cmap) 
            plt.colorbar()
            plt.ylim(top=0, bottom =-240) 
            plt.xlim(left=0, right = 90)
            ax = plt.gca() 
            plt.ylabel('Depth [m]')
            plt.xlabel('Time [days]') 

            ''' using colorbar from above to separate colours by initial release latitudinal bins '''
            
            ax = fig.add_subplot(gs[r,c])
            
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
                    
                ax.plot(time_p[ii,boo_p],(depths_p[ii,boo_p]*-1), c = rgb, linewidth=1, alpha = 0.6)
                ind_nonan = np.where(depths_p[ii,boo_p]>0.6)
                if np.array(ind_nonan).any():
                    last_ind = ind_nonan[0][-1]
                    ax.plot(time_p[ii,last_ind],(depths_p[ii,last_ind]*-1), marker = 'o', c = rgb, markersize=5, linewidth = 0, markeredgecolor='black',alpha = 0.6) 
                
                    time_save[ii] = time_p[ii,last_ind]
                    depths_save[ii] = depths_p[ii,last_ind]
                    lats_save[ii] = lats_p[ii,0]

            ax.set_ylim(top=0, bottom =-240) 
            ax.set_xlim(left=0, right = 90)

            plt.rc('font') 
            if row == 2 and col ==0:
                ax.set_ylabel('Depth [m]') 
            
            if row ==4 and col == 1:
                ax.set_xlabel('Time [days]') 
                
            if row<4: 
                ax.set_xticklabels([])

            if col>0:
                ax.set_yticklabels([])
                
            if col == 0:
                ax.text(-0.4, 0.5, f'radius = \n{size_name}', va='bottom', ha='center',
                    rotation='horizontal', rotation_mode='anchor',
                    transform=ax.transAxes)

            if row == 0: 
                ax.title.set_text(rf'$\rho$ = {rho} kg m$^-$$^3$'+f'\n({letter})') #' radius = '+size_name) #$T_s$ 
            else:
                ax.title.set_text(f'({letter})')   

        
            if row == 4:
                bounds = [-50,-30,-10,10,30,50]
                cbaxes = fig.add_axes([0.2, 0, 0.75, 0.01])
                cbar = plt.colorbar(scat, cax=cbaxes, orientation="horizontal", aspect=50, extend='both', ticks = bounds, label = '[initial latitude]')
        
        '''UNCOMMENT THIS TO GET EXACT VALUES FOR MEDIAN TS AND TIME TO REACH ZS TO ADD TO MANUSCRIPT '''
        # Ts[Ts==100] = np.nan
        # print(f'Median Ts for global {size_name} particles = '+str(np.nanmedian(Ts.ravel().data)))

#%%

'''' Uncomment to save variables - NB now I've converted NaN (no sinking within 90 days) to 100 days.'''       
# with open(dirwrite+'90day_global_Ts_for03and07.pickle', 'wb') as f:
#     pickle.dump([lons,lats,Ts03,Ts07], f)

# with open(dirwrite+'90day_rho'+rho+'_global_Ts_allsizes.pickle', 'wb') as f: # 30th Nov, now saving for all sizes to be used for 3-yr sims 
#     pickle.dump([lons,lats,Ts03,Ts04,Ts05,Ts06,Ts07], f)

# with open(dirwrite+'Ts2004_allsizes_for_diffplot_with_advection.pickle', 'wb') as f:
#     pickle.dump([lons,lats,Ts02,Ts03,Ts04,Ts05,Ts06,Ts07], f)

'''Save the figure from the Spyder plot panel, top-right'''
