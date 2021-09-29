import xarray as xr
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats
import statsmodels.api as sm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from matplotlib.patches import Ellipse
from plot_potential_impact import get_myjet_cmap 

# Geometric mean regression; Ref: Least Squares Regression vs. Geometric Mean Regression for Ecotoxicology Studies (1995)
# b,a,r=geometric_mean_regression(x,y) b:slope, a:const, r: correlation and p value
def geometric_mean_regression(XY):
    x = XY[:,0]
    y = XY[:,1]
    X = sm.add_constant(x)
    model_x = sm.OLS(y, X, missing='drop').fit()
    X2 = sm.add_constant(y)
    model_y = sm.OLS(x, X2, missing='drop').fit()
    b = (model_x.params[1]/model_y.params[1])**0.5
    a = np.nanmean(y) - np.nanmean(x) * b
    r=scipy.stats.spearmanr(x, y)#pearsonr
    print('correlation is %f and p is %f'%(r[0],r[1]))
    return b,a,r

def f(p, x):
    """Basic linear regression 'model' for use with ODR"""
    return (p[0] * x) + p[1]

# Process fluxnet data from collaborator, 
# extract cloud effect for paried sites and assign region label 
def load_flux_data(rerun=False,remove_outlier=True):
    if rerun:
        cloud=xr.open_dataset('../data/results/xu/potential_one_deg_0207.nc')# 1deg cloud effect
        flux_h=pd.read_csv('../data/H_diff.csv') # Add two pair amazon flux sites
        # Add mean lat/lon for paired sites
        flux_h.loc[:,'lat_mean']=flux_h.loc[:,['pair_open_lat','pair_forest_lat']].mean(axis=1)
        flux_h.loc[:,'lon_mean']=flux_h.loc[:,['pair_open_lon','pair_forest_lon']].mean(axis=1)
        # Add continent label 
        ind=(flux_h['lon_mean']>-130) & (flux_h['lon_mean']<-60)&(flux_h['lat_mean']>0) & (flux_h['lat_mean']<65)
        flux_h.loc[ind,'region']='NoA'
        ind=(flux_h['lon_mean']>0) & (flux_h['lon_mean']<60)&(flux_h['lat_mean']>30) & (flux_h['lat_mean']<70)
        flux_h.loc[ind,'region']='EU'
        flux_h.loc[14,'region']='AU'
        flux_h.loc[28:29,'region']='AM'
        # append delta cloud value 
        temp = [cloud.potential.sel(lat=flux_h['lat_mean'][i], lon=flux_h['lon_mean'][i], method='nearest').values for i in range(flux_h.shape[0])]
        flux_h.loc[:,'cloud_diff'] =np.array(temp)

        # Group sites in close distance for plotting
        c = [1] + [2] + [3]*2 + [5]*4 + [9]*2 + [11] +[12]*3 + [15] + [16]*2 + [18,19,20] + [16,22] +[3]*2 + [25]*2 + [27,28,29,30]
        flux_h.loc[:,'group']=c
        # Add group number to each site pair
        flux_h=flux_h.join(flux_h.groupby('group')['pair'].count().rename('group_num'), on='group')
        # Add lat/lon offset to each pair for display purpose
        flux_h.loc[:,'lat_plotting']=np.nan
        flux_h.loc[:,'lon_plotting']=np.nan
        v_offset = np.array([[-2, 0],[0,2], [0,-2], [2,0]])
        for i in flux_h.group.unique():
            ind = (flux_h.loc[:,'group']==i)
            if ind.sum()==1:
                flux_h.loc[ind,['lon_plotting','lat_plotting']]=flux_h.loc[ind,['lon_mean','lat_mean']].values
            else: 
                flux_h.loc[ind,['lon_plotting','lat_plotting']]=flux_h.loc[ind,['lon_mean','lat_mean']].values + v_offset[0:ind.sum() ,:]  

        flux_h.loc[:,'dif'] = -flux_h.loc[:,'dif']
        flux_h.to_csv('../data/results/my_flux_h0207.csv')
        print('flux data csv saved')
    else:
        flux_h = pd.read_csv('../data/results/my_flux_h0207.csv')
        print('read csv data from saved file')
    if remove_outlier:
        flux_h = flux_h[(flux_h.dif<150)&(flux_h.dif>-100)]
    return flux_h

# Add ellipse for key cloud inhibition region
def add_circle(ax):
    # Amazon
    e1 = Ellipse(xy=(-60, -15),
                    width=50, height=25,
                    angle=25, edgecolor='k', facecolor='None',ls='--')
    # southeast USA
    e2 = Ellipse(xy=(-90, 35),
                    width=35, height=20,
                    angle=0, edgecolor='k', facecolor='None',ls='--')
    # Africa
    e3 = Ellipse(xy=(20, 0),
                    width=40, height=30,
                    angle=0, edgecolor='k', facecolor='None',ls='--')

    ax.add_artist(e1)
    ax.add_artist(e2)
    ax.add_artist(e3)

def make_plot(rerun=False):
    cloud05=xr.open_dataset('../data/results/xu/MODIS_potential_0207.nc')
    msg14=xr.open_dataset('../data/results/xu/MSG_potential0908.nc')
    h_sa=xr.open_dataset('../data/results/xu/HG_det.nc')
    h_clm=xr.open_dataset('../data/results/xu/CLM_SH.nc')
    flux_h=load_flux_data(rerun=rerun)
    mycmap=get_myjet_cmap(levels=np.arange(-0.15,0.151,0.025)) # use levels for color
    mycmap_flux=get_myjet_cmap(levels=np.arange(-100,101,25)) # use levels for color

    fig = plt.figure(figsize=[10,6])

    # Cloud effect panel
    pos1 = [0.05, 0.725, 0.5, 0.5] # [left, bottom, width, height]
    ax1 = fig.add_axes(pos1, projection=ccrs.PlateCarree())
    cloud05.potential.plot(cmap=mycmap, vmin=-0.15,vmax=0.15, ax=ax1, add_colorbar=False, rasterized=True) 
    ax1.set_extent([-180, 180, -60, 80])
    ax1.coastlines()
    ax1.set_title('Potential cloud effect ($\Delta$Cloud)')
#    ax1.text(0.015, 0.6, '$\Delta$Cloud', fontsize=12,transform=ax1.transAxes)
    ax1.text(0.5, 0.05, 'MODIS', fontsize=12,transform=ax1.transAxes,ha='center',fontweight='bold')
    add_circle(ax1)

    # MSG inset Panel
    pos1in = [0.035, 0.825, 0.15, 0.15] # [left, bottom, width, height]
    ax1in = fig.add_axes(pos1in, projection=ccrs.PlateCarree())
    msg14.potential.plot(cmap=mycmap, ax=ax1in, add_colorbar=False, rasterized=True)
    ax1in.set_extent([-70, 60, -20, 45])
    ax1in.coastlines()
    ax1in.text(0.25, 0.05, 'MSG', fontsize=12,transform=ax1in.transAxes,fontweight='bold')
    add_circle(ax1in)

    # Add colorbar for cloud effect
    cbar1_pos = [ax1.get_position().x0-0.02, ax1.get_position().y0, 0.01,ax1.get_position().height]
    cax1 = fig.add_axes(cbar1_pos)
    cb1 = mpl.colorbar.ColorbarBase(ax=cax1, cmap=mycmap, norm=Normalize(vmin=-0.15, vmax=0.15) ,
                                                    orientation='vertical',ticks=np.arange(-0.15,0.16,0.05)) 
    cb1.ax.set_yticklabels([-0.15,-0.1,-0.05 ,0, 0.05,0.10, 0.15], fontsize=9)
    cax1.tick_params(axis="y",direction='out', left=True, labelleft=True, right=False, labelright=False, pad=0)
    # cb1.set_label('$\Delta$Cloud', fontsize=10, labelpad=0)

    ## Satellite panel
    pos2 = [0.05, 0.325, 0.5, 0.5] # [left, bottom, width, height]
    ax2 = fig.add_axes(pos2, projection=ccrs.PlateCarree())
    (-h_sa.HG_det).plot(cmap=mycmap,vmin=-50,vmax=50, ax=ax2, add_colorbar=False, rasterized=True) # tab10, set3

    # ax2.set_position([ax2.get_position().x0-0.05, ax2.get_position().y0, ax1.get_position().width , ax1.get_position().height])

    ax2.set_extent([-180, 180, -60, 80])
    ax2.coastlines()
    ax2.set_title('Sensible heat difference ($\Delta$H)')
#    ax2.text(0.025, 0.05, '$\Delta$H', fontsize=12,transform=ax2.transAxes)
    ax2.text(0.5, 0.05, 'Satellite', fontsize=12,transform=ax2.transAxes ,ha='center',fontweight='bold')
    add_circle(ax2)

    ## CLM panel
    pos3 = [0.05, 0.0, 0.5, 0.5] # [left, bottom, width, height]
    ax3 = fig.add_axes(pos3, projection=ccrs.PlateCarree())
    (-h_clm.SH).plot(cmap=mycmap,vmin=-50,vmax=50, ax=ax3, add_colorbar=False, rasterized=True) 
    ax3.set_extent([-180, 180, -60, 80])
    ax3.coastlines()
#    ax3.text(0.025, 0.05, '$\Delta$H', fontsize=12,transform=ax3.transAxes)
    ax3.text(0.5, 0.05, 'CLM', fontsize=12,transform=ax3.transAxes, ha='center', fontweight='bold')
    ax3.text(-0.05, 0.05, '$W/m^2$', fontsize=10,transform=ax3.transAxes, ha='center')
    add_circle(ax3)

    # Add colorbar for CLM
    cbar3_pos = [ax3.get_position().x0-0.02,
                 ax3.get_position().y0+(ax3.get_position().height+ax2.get_position().height)*0.1,
                 0.01, 
                 (ax3.get_position().height+ax2.get_position().height)*0.8]
    cax3 = fig.add_axes(cbar3_pos)
    cb3 = mpl.colorbar.ColorbarBase(ax=cax3, cmap=mycmap, norm=Normalize(vmin=-50, vmax=50) ,
                                                    orientation='vertical',ticks=np.arange(-50,51,10))
    cb3.ax.set_yticklabels(np.arange(-50, 51,10), fontsize=9)
    cax3.tick_params(axis="y",direction='out', left=True, labelleft=True, right=False, labelright=False, pad=0)

    ##  Flux location 1  panel: NA
    pos4 = [0.6, 0.8, 0.25, 0.25] # [left, bottom, width, height]
    ax4 = fig.add_axes(pos4,projection=ccrs.PlateCarree())
    ind1=(flux_h['group_num']>1)
#    ind=(flux_h['lon_mean']>-130) & (flux_h['lon_mean']<-60)&(flux_h['lat_mean']>0) & (flux_h['lat_mean']<65)
    ind=flux_h['region']=='NoA'

    # Plot connecting lines
    for i in flux_h[ind1&ind].group.unique():
        ind0 = (flux_h.loc[:,'group']==i)
        ax4.plot([flux_h.loc[ind0,'lon_mean'],flux_h.loc[ind0,'lon_plotting']],[flux_h.loc[ind0,'lat_mean'],flux_h.loc[ind0,'lat_plotting']]
                             ,color='k',lw=0.5)

    # Plot H at plotting lat/lon    
    ax4.scatter(flux_h.loc[ind,'lon_plotting'], flux_h.loc[ind,'lat_plotting'], c=flux_h.loc[ind,'dif'], s=20,marker='o',cmap=mycmap,vmax=100, vmin=-100)
    # plot cluster center location  
    ax4.scatter(flux_h.loc[ind1,'lon_mean'], flux_h.loc[ind1,'lat_mean'], s=10, marker='.',color='k')
    # ax4.set_extent([-130, -70, 20, 60])
    ax4.set_extent([-125, -70, 30, 60])
    ax4.coastlines()
    
    ax4.add_feature(cfeature.LAND, facecolor=[0.88,0.88,0.88]) #, 'lightgrey'
    # ax4.add_feature(cfeature.LAKES,edgecolor='black',facecolor='None') # Add lake
    ax4.add_feature(cfeature.BORDERS, linestyle=':')
    # ax4.set_title('North America')
    ax4.text(0.05, 0.9, 'North America', fontsize=10,transform=ax4.transAxes)
    
    ax4.text(0.5, 0.05, 'Paired flux site', fontsize=12, transform=ax4.transAxes, ha='center', fontweight='bold')
#    ax4.text(0.01, 0.035, '$\Delta$H', fontsize=12,transform=ax4.transAxes)
    ax4.text(0.75, 1.075, 'Sensible heat difference ($\Delta$H)', fontsize=12, transform=ax4.transAxes, ha='center')
    
    ##  Flux location 2  panel: EU
    pos5 = [0.8, 0.8, 0.25, 0.25] # [left, bottom, width, height]
    ax5 = fig.add_axes(pos5,projection=ccrs.PlateCarree())
 #   ind=(flux_h['lon_mean']>0) & (flux_h['lon_mean']<60)&(flux_h['lat_mean']>30) & (flux_h['lat_mean']<70)
    ind=flux_h['region']=='EU'

    # Plot connecting lines
    for i in flux_h[ind1&ind].group.unique():
        ind0 = (flux_h.loc[:,'group']==i)
        ax5.plot([flux_h.loc[ind0,'lon_mean'],flux_h.loc[ind0,'lon_plotting']],[flux_h.loc[ind0,'lat_mean'],flux_h.loc[ind0,'lat_plotting']]
                  ,color='k',lw=0.5)

    # Plot H at plotting lat/lon    
    ax5.scatter(flux_h.loc[ind,'lon_plotting'], flux_h.loc[ind,'lat_plotting'], c=flux_h.loc[ind,'dif'], s=20,marker='o',cmap=mycmap_flux,vmax=100, vmin=-100)
    # plot cluster center location  
    ax5.scatter(flux_h.loc[ind1,'lon_mean'], flux_h.loc[ind1,'lat_mean'], s=10, marker='.',color='k')
    ax5.set_extent([0, 20, 40, 60])
    ax5.coastlines()
    # ax4.add_feature(cfeature.OCEAN)
    ax5.add_feature(cfeature.LAND, facecolor=[0.88,0.88,0.88]) #, 'lightgrey'
    ax5.add_feature(cfeature.BORDERS, linestyle=':')
    # ax5.set_title('Europe')
    ax5.text(0.025, 0.8, 'Europe', fontsize=10,transform=ax5.transAxes)

    # Colorbar for flux tower map
    cbar5_pos = [ax4.get_position().x0 +(ax4.get_position().width+ax5.get_position().width)*0.1 ,
                 ax5.get_position().y0-0.025, 0.8*(ax4.get_position().width+ax5.get_position().width), 0.015]
    cax5 = fig.add_axes(cbar5_pos)
    cb5 = mpl.colorbar.ColorbarBase(ax=cax5, cmap=mycmap_flux, norm=Normalize(vmin=-100, vmax=100) ,
                                    orientation='horizontal', ticks=np.arange(-100,101,25))
    cb5.ax.set_xticklabels(np.arange(-100, 101,25), fontsize=9)
    ax5.text(0.9, -0.125, '$W/m^2$', fontsize=10,transform=ax5.transAxes, ha='center')

    ##  Flux H and cloud effect panel
    # Estimate regression line
    ind=flux_h['dif']>-200

    p = geometric_mean_regression(flux_h.loc[ind,['dif','cloud_diff']].dropna().values)

    pos6 = [0.65, 0.15, 0.35, 0.5] # [left, bottom, width, height]
    ax6 = fig.add_axes(pos6)

    with sns.axes_style("ticks"):
        sns.scatterplot(x="dif", y="cloud_diff", data=flux_h,hue='region',ax=ax6)
    ax6.legend(ax6.get_legend_handles_labels()[0][1::], ['Europe','North America','Australia','Amazon'],frameon=False)

    ax6.plot(np.arange(-100,130,1), f(p[0:2], np.arange(-100,130,1)),color='r')
    ax6.plot([-100,130],[0,0],'--',lw=0.5,color='grey')
    ax6.plot([0,0],[-0.06,0.06],'--',lw=0.5,color='grey')
    ax6.set_xlim([-100,130])
    ax6.set_ylim([-0.06,0.06])
    ax6.set_ylabel('$\Delta$Cloud', labelpad=0)
    ax6.set_xlabel('$\Delta$H ($W/m^2$)')
    ax6.set_title('Paired Flux site')
    ax6.text(0.1, 0.1, r'$\rho$=%.2f, $\it{p}$=%.2f'%(p[2][0],p[2][1]), fontsize=10, transform=ax6.transAxes)

    ax1.text(-0.02, 1.05, 'a', fontsize=14, transform=ax1.transAxes, fontweight='bold')
    ax2.text(-0.02, 1.05, 'b', fontsize=14, transform=ax2.transAxes, fontweight='bold')
    ax4.text(-0.02, 1.05, 'c', fontsize=14, transform=ax4.transAxes, fontweight='bold')
    ax6.text(-0.02, 1.05, 'd', fontsize=14, transform=ax6.transAxes, fontweight='bold')

   # plt.savefig('../figure/figure_sensible_heat0702.png',dpi=300,bbox_inches='tight')
    plt.savefig('../figure/figure3.pdf',bbox_inches='tight')
    print('figure saved')

if __name__=='__main__':
    make_plot(rerun=False)
#    flux_h=load_flux_data(rerun=True) # reprocess flux sensible heat data for plotting
