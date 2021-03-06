#import pandas as pd
import matplotlib.pyplot as plt
#from matplotlib.colors import ListedColormap, Normalize
from matplotlib.colors import Normalize
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
#import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib as mpl
import statsmodels.api as sm
from plot_figure1 import get_myjet_cmap

def norm_cmap(cmap, vmin=None, vmax=None):
    """
    Normalize and set colormap

    Parameters
    ----------
    values : Series or array to be normalized
    cmap : matplotlib Colormap
    normalize : matplotlib.colors.Normalize
    cm : matplotlib.cm
    vmin : Minimum value of colormap. If None, uses min(values).
    vmax : Maximum value of colormap. If None, uses max(values).

    Returns
    -------
    n_cmap : mapping of normalized values to colormap (cmap)

    """
#     mn = vmin or min(values)
#     mx = vmax or max(values)
#     norm = Normalize(vmin=mn, vmax=mx)
    norm = Normalize(vmin=vmin, vmax=vmax)
    n_cmap = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    return n_cmap, norm

def plot_pcolor_map(ds, ax, vminmax,cmap,add_colorbar=False):
    if vminmax:
        ds.plot.pcolormesh(ax=ax,transform=ccrs.PlateCarree(), add_colorbar=add_colorbar, cmap=cmap,vmin=vminmax[0],vmax=vminmax[1])
    else:
        ds.plot.pcolormesh(ax=ax,transform=ccrs.PlateCarree(), add_colorbar=add_colorbar, cmap=cmap)
    ax.coastlines()

def plot_region_box(ax, latlon):
    x, y = [latlon[2], latlon[2], latlon[3], latlon[3], latlon[2]], [latlon[0], latlon[1], latlon[1], latlon[0], latlon[0]] # Amazon
    ax.plot(x, y, transform=ccrs.PlateCarree(),color='k',linewidth=0.75)

def plot_subplot_label(ax, txt, left_offset=-0.05):
    ax.text(left_offset, 1, txt, fontsize=14, transform=ax.transAxes, fontweight='bold')

def plot_region_label(ax, txt, x,y):
    ax.text(x,y,txt ,transform=ax.transAxes)

def plot_yearly_change(ds,ax):
    (100*ds.mean(dim=['lat','lon'])).plot(ax=ax,color='b')
    ax.set_xlabel('')
    ax.set_xticklabels('')
    ax.set_ylabel(r'$\Delta$Cloud$\times$100')
    ax.tick_params(axis='y', which='both', direction='out', right=True, labelright=True, left=False,labelleft=False)
#     ax.yaxis.set_label_position("right")

def plot_trend_line(ax, ds):
    # Trend line
    ax.plot(range(2002,2019,1),linear_fit(100*ds.mean(dim=['lat','lon'])),color='r')


def set_lat_lon(ax, xtickrange, ytickrange, label=False,pad=0.05, fontsize=8):
    lon_formatter = LongitudeFormatter(zero_direction_label=True, degree_symbol='')
    lat_formatter = LatitudeFormatter(degree_symbol='')
    ax.set_yticks(ytickrange, crs=ccrs.PlateCarree())
    ax.set_xticks(xtickrange, crs=ccrs.PlateCarree())
    if label:
        ax.set_xticklabels(xtickrange,fontsize=fontsize)
        ax.set_yticklabels(ytickrange,fontsize=fontsize)
        ax.tick_params(axis='x', which='both', direction='out', bottom=True, top=False,labeltop=False,labelbottom=True, pad=pad)
        ax.tick_params(axis='y', which='both', direction='out', pad=pad)

    else:
        ax.tick_params(axis='x', which='both', direction='out', bottom=True, top=False, labeltop=False, labelleft=False, labelbottom=False)
        ax.tick_params(axis='y', which='both', direction='out', left=True, labelleft=False)

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_ylabel('')
    ax.set_xlabel('')

def linear_fit(ds):
    X = sm.add_constant(ds.time)
    model = sm.OLS(ds.values, X)
    results = model.fit()
    return results.predict()


def make_plot():
    # Load data
    loss_yearly=xr.open_dataset('../data/results/xu/forest_loss_effect_year.nc')
    floss=xr.open_dataset('../data/results/forestloss_2018_05deg.nc')
    trend=xr.open_dataset('../data/results/xu/forest_loss_impact_based_on_trend.nc')
    ds05=xr.open_dataset('../data/results/xu/result.nc')

    # jet cmap
    mycmap=get_myjet_cmap()

    fig = plt.figure(figsize=[10,10])
    
    fig, axs = plt.subplots(ncols=4, nrows=6,figsize=[12,12])
    gs = axs[1, 2].get_gridspec()
    # remove the underlying axes
    for ax in axs[0:6, 0:4].flatten():
        ax.remove()
    # https://matplotlib.org/3.1.3/gallery/subplots_axes_and_figures/gridspec_and_subplots.html    
    axbig1 = fig.add_subplot(gs[0:2, 0:2],projection=ccrs.PlateCarree())
    axbig2 = fig.add_subplot(gs[0:2, 2:4],projection=ccrs.PlateCarree())
    
    floss.forest_loss.plot(ax=axbig1,cmap='hot',rasterized=True, add_colorbar=False)
    ds05.loss.plot(ax=axbig2,cmap=mycmap,vmin=-0.15, vmax=0.15,rasterized=True, add_colorbar=False)
    axbig1.set_extent([-180, 180, -60, 80])
    axbig2.set_extent([-180, 180, -60, 80])
    axbig1.coastlines()
    axbig2.coastlines()
    
    # Move big plot up 
    axbig1.set_position([axbig1.get_position().x0, axbig1.get_position().y0+0.035, axbig1.get_position().width,axbig1.get_position().height])
    axbig2.set_position([axbig2.get_position().x0, axbig2.get_position().y0+0.035, axbig2.get_position().width,axbig2.get_position().height])
    
    # Plot region boundary 
    plot_region_box(axbig1, [-20,0,-70,-40]) # Amazon
    plot_region_box(axbig1, [-5,15,95,125]) # Indonesia
    plot_region_box(axbig1, [55,70,115,137.5]) # East Seberia
    plot_region_box(axbig1, [25,40,-95,-72.5]) # SouthEast USA
    
    plot_region_box(axbig2, [-20,0,-70,-40]) # Amazon
    plot_region_box(axbig2, [-5,15,95,125]) # Indonesia
    plot_region_box(axbig2, [55,70,115,137.5]) # East Seberia
    plot_region_box(axbig2, [25,40,-95,-72.5]) # SouthEast USA
    
    axbig1.set_title('Forest loss', fontsize=14)
    axbig2.set_title('Forest loss impact on cloud',fontsize=14)
    
    # Add colorbar to big plot
    cbarbig1_pos = [axbig1.get_position().x0, axbig1.get_position().y0-0.025, axbig1.get_position().width, 0.01]
    caxbig1 = fig.add_axes(cbarbig1_pos)
    cmap0, norm = norm_cmap(cmap='hot', vmin=-0.8, vmax=0)
    cbbig1 = mpl.colorbar.ColorbarBase(ax=caxbig1, cmap=cmap0.cmap, norm=norm,orientation='horizontal', ticks=np.arange(-0.8, 0, 0.2)) #cmap=plt.get_cmap('hot')
    cbbig1.ax.set_yticklabels(np.arange(-0.8, 0.1,0.2),fontsize=10)
    cbbig1.set_label('Forest loss fraction', fontsize=12)
    
    cbarbig2_pos = [axbig2.get_position().x0, axbig2.get_position().y0-0.025, axbig2.get_position().width, 0.01]
    caxbig2 = fig.add_axes(cbarbig2_pos)
    # cmap0, norm = norm_cmap(cmap='hot', vmin=-0.8, vmax=0)
    cbbig2 = mpl.colorbar.ColorbarBase(ax=caxbig2, cmap=mycmap, norm=Normalize(vmin=-0.25, vmax=0.25),
                                       orientation='horizontal', ticks=np.arange(-0.2, 0.21, 0.1)) 
    cbbig2.ax.set_yticklabels(np.arange(-0.2, 0.21, 0.1),fontsize=10)
    cbbig2.set_label('$\Delta$Cloud', fontsize=12)
    
    # Regional zoomed map
    ax11 = fig.add_subplot(6, 4, 1+8, projection=ccrs.PlateCarree())
    ax12 = fig.add_subplot(6, 4, 2+8, projection=ccrs.PlateCarree())
    ax13 = fig.add_subplot(6, 4, 3+8, projection=ccrs.PlateCarree())
    ax14 = fig.add_subplot(6, 4, 4+8)
    
    ax21 = fig.add_subplot(6, 4, 5+8, projection=ccrs.PlateCarree())
    ax22 = fig.add_subplot(6, 4, 6+8, projection=ccrs.PlateCarree())
    ax23 = fig.add_subplot(6, 4, 7+8, projection=ccrs.PlateCarree())
    ax24 = fig.add_subplot(6, 4, 8+8)
    
    ax31 = fig.add_subplot(6, 4, 9+8, projection=ccrs.PlateCarree())
    ax32 = fig.add_subplot(6, 4, 10+8, projection=ccrs.PlateCarree())
    ax33 = fig.add_subplot(6, 4, 11+8, projection=ccrs.PlateCarree())
    ax34 = fig.add_subplot(6, 4, 12+8)
    
    ax41 = fig.add_subplot(6, 4, 13+8, projection=ccrs.PlateCarree())
    ax42 = fig.add_subplot(6, 4, 14+8, projection=ccrs.PlateCarree())
    ax43 = fig.add_subplot(6, 4, 15+8, projection=ccrs.PlateCarree())
    ax44 = fig.add_subplot(6, 4, 16+8)
    
    # Amazon
    plot_pcolor_map(floss.forest_loss.where(floss.forest_loss!=0).loc[-20:0,-70:-40], ax=ax11,vminmax=[-0.8,0], cmap='hot')
    plot_pcolor_map(ds05.loss.where(floss.forest_loss<-0.05).loc[-20:0,-70:-40], ax=ax12, vminmax=[-0.15,0.15], cmap=mycmap)
    # plot_pcolor_map(loss_yearly.forest_loss_effect.where(floss.forest_loss<-0.05).loc[2014:2018,-20:0,-70:-40].mean(dim='time'), 
    #                                 ax=ax12, vminmax=[-0.15,0.15], cmap=mycmap)
    plot_pcolor_map(trend.loss_impact.where(floss.forest_loss<-0.05).loc[-20:0,-70:-40], ax=ax13, vminmax=[-0.015,0.015], cmap=mycmap)
    plot_yearly_change(loss_yearly.forest_loss_effect.where(floss.forest_loss<-0.05).loc[2002:2018,-20:0,-70:-40],ax=ax14)
    plot_trend_line(ax14, loss_yearly.forest_loss_effect.where(floss.forest_loss<-0.05).loc[2002:2018,-20:0,-70:-40])
    
    set_lat_lon(ax11, range(-70,-39,10), range(-20,1,10), label=True)
    set_lat_lon(ax12, range(-70,-39,10), range(-20,1,10))
    set_lat_lon(ax13, range(-70,-39,10), range(-20,1,10))
    
    # Indonisia
    plot_pcolor_map(floss.forest_loss.where(floss.forest_loss!=0).loc[-5:15,95:125], ax=ax21,vminmax=[-0.8,0], cmap='hot')
    plot_pcolor_map(ds05.loss.where(floss.forest_loss<-0.05).loc[-5:15,95:125], ax=ax22, vminmax=[-0.15,0.15], cmap=mycmap)
    # plot_pcolor_map(loss_yearly.forest_loss_effect.where(floss.forest_loss<-0.05).loc[2014:2018,-5:15,95:125].mean(dim='time'), 
    #                                 ax=ax22, vminmax=[-0.15,0.15], cmap=mycmap)
    plot_pcolor_map(trend.loss_impact.where(floss.forest_loss<-0.05).loc[-5:15,95:125], ax=ax23, vminmax=[-0.015,0.015], cmap=mycmap)
    plot_yearly_change(loss_yearly.forest_loss_effect.where(floss.forest_loss<-0.05).loc[2002:2018,-5:15,95:125], ax=ax24)
    plot_trend_line(ax24, loss_yearly.forest_loss_effect.where(floss.forest_loss<-0.05).loc[2002:2018,-5:15,95:125])
    
    set_lat_lon(ax21, range(95,126,10), range(-5,14,10), label=True)
    set_lat_lon(ax22, range(95,126,10), range(-5,14,10))
    set_lat_lon(ax23, range(95,126,10), range(-5,14,10))
    
    # East Siberia
    plot_pcolor_map(floss.forest_loss.where(floss.forest_loss!=0).loc[55:70,115:137.5], ax=ax31,vminmax=[-0.8,0], cmap='hot')
    plot_pcolor_map(ds05.loss.where(floss.forest_loss<-0.05).loc[55:70,115:137.5], ax=ax32, vminmax=[-0.15,0.15], cmap=mycmap)
    # plot_pcolor_map(loss_yearly.forest_loss_effect.where(floss.forest_loss<-0.05).loc[2014:2018,55:70,115:137.5].mean(dim='time'), 
    #                                 ax=ax32, vminmax=[-0.15,0.15], cmap=mycmap)
    plot_pcolor_map(trend.loss_impact.where(floss.forest_loss<-0.05).loc[55:70,115:137.5], ax=ax33, vminmax=[-0.015,0.015], cmap=mycmap)
    plot_yearly_change(loss_yearly.forest_loss_effect.where(floss.forest_loss<-0.05).loc[2002:2018,55:70,115:137.5], ax=ax34)
    plot_trend_line(ax34, loss_yearly.forest_loss_effect.where(floss.forest_loss<-0.05).loc[2002:2018,55:70,115:137.5])
    
    set_lat_lon(ax31, range(115,137,10), range(55,71,10), label=True)
    set_lat_lon(ax32, range(115,137,10), range(55,71,10))
    set_lat_lon(ax33, range(115,137,10), range(55,71,10))
    
    #   Southeast America
    plot_pcolor_map(floss.forest_loss.where(floss.forest_loss!=0).loc[25:40,-95:-72.5], ax=ax41,vminmax=[-0.8,0], cmap='hot')
    plot_pcolor_map(ds05.loss.where(floss.forest_loss<-0.05).loc[25:40,-95:-72.5], ax=ax42, vminmax=[-0.15,0.15], cmap=mycmap)
    # plot_pcolor_map(loss_yearly.forest_loss_effect.where(floss.forest_loss<-0.05).loc[2014:2018,25:40,-95:-72.5].mean(dim='time'), 
    #                                 ax=ax42, vminmax=[-0.15,0.15], cmap=mycmap)
    plot_pcolor_map(trend.loss_impact.where(floss.forest_loss<-0.05).loc[25:40,-95:-72.5], ax=ax43, vminmax=[-0.015,0.015], cmap=mycmap)
    plot_yearly_change(loss_yearly.forest_loss_effect.where(floss.forest_loss<-0.05).loc[2002:2018,25:40,-95:-72.5], ax=ax44)
    plot_trend_line(ax44, loss_yearly.forest_loss_effect.where(floss.forest_loss<-0.05).loc[2002:2018,25:40,-95:-72.5])
    
    set_lat_lon(ax41, range(-95,-73,10), range(25,41,10), label=True)
    set_lat_lon(ax42, range(-95,-73,10), range(25,41,10))
    set_lat_lon(ax43, range(-95,-73,10), range(25,41,10))
    
    # Fix panel D time series axis size
    ax14.set_position([ax14.get_position().x0,ax11.get_position().y0, ax11.get_position().width,ax11.get_position().height])
    ax24.set_position([ax24.get_position().x0,ax21.get_position().y0, ax21.get_position().width,ax21.get_position().height])
    ax34.set_position([ax34.get_position().x0,ax31.get_position().y0, ax31.get_position().width,ax31.get_position().height])
    ax44.set_position([ax44.get_position().x0,ax41.get_position().y0, ax41.get_position().width,ax41.get_position().height])
    
    # Same year label for panel 4
    ax14.set_xticks(range(2002, 2019, 4))
    ax14.set_xticklabels('')
    ax44.set_xticks(range(2002, 2019, 4))
    ax44.set_xticklabels(range(2002, 2019,4))
    ax44.set_xlabel('Year')
    
    # Panel title
    ax11.set_title('Forest loss')
    ax12.set_title('Mean cloud impact')
    ax13.set_title('Cloud impact trend')
    ax14.set_title('Regional cloud impact')
    
    # Add region names in column 4
    ax14.text(0.05,0.88,'Amazon' ,transform=ax14.transAxes)
    ax24.text(0.05,0.55,'Indonesia' ,transform=ax24.transAxes)
    ax34.text(0.05,0.05,'East Siberia' ,transform=ax34.transAxes)
    ax44.text(0.2,0.1,'Southeast US' ,transform=ax44.transAxes)
    
    # for i in [ax11, ax12, ax13] plot_region_label(i, 'Amazon', 0.05,0.85)
    [plot_region_label(i, 'Amazon', 0.04,0.88) for i in [ax11, ax12, ax13]]
    [plot_region_label(i, 'Indonesia', 0.3,0.55) for i in [ax21, ax22, ax23]]
    [plot_region_label(i, 'East Siberia', 0.55,0.88) for i in [ax31, ax32, ax33]]
    [plot_region_label(i, 'Southeast US', 0.04,0.1) for i in [ax41, ax42, ax43]]
    
    # Colorbar for regional plot
    cbar41_pos = [ax41.get_position().x0, ax41.get_position().y0-0.035, ax41.get_position().width, 0.01]
    cax41 = fig.add_axes(cbar41_pos)
    cmap0, norm = norm_cmap(cmap='hot', vmin=-0.8, vmax=0)
    cb41 = mpl.colorbar.ColorbarBase(ax=cax41, cmap=cmap0.cmap, norm=norm,orientation='horizontal', ticks=np.arange(-0.8, 0, 0.2)) #cmap=plt.get_cmap('hot')
    cb41.ax.set_yticklabels(np.arange(-0.8, 0.1,0.2),fontsize=10)
    cb41.set_label('Forest loss fraction', fontsize=12)
    
    cbar42_pos = [ax42.get_position().x0, ax42.get_position().y0-0.035, ax42.get_position().width, 0.01]
    cax42 = fig.add_axes(cbar42_pos)
    cb42 = mpl.colorbar.ColorbarBase(ax=cax42, cmap=mycmap, norm=Normalize(vmin=-0.25, vmax=0.25),
                                     orientation='horizontal', ticks=np.arange(-0.2, 0.21, 0.1)) #cmap=plt.get_cmap('hot')
    cb42.set_label('$\Delta$Cloud', fontsize=12)
    
    cbar43_pos = [ax43.get_position().x0, ax43.get_position().y0-0.035, ax43.get_position().width, 0.01]
    cax43 = fig.add_axes(cbar43_pos)
    cb43 = mpl.colorbar.ColorbarBase(ax=cax43, cmap=mycmap, norm=Normalize(vmin=-0.015, vmax=0.015),
                                     orientation='horizontal', ticks=np.arange(-0.015, 0.016, 0.005)) #cmap=plt.get_cmap('hot')
    cb43.set_ticklabels(np.arange(-1.5, 1.6, 0.5))
    cb43.set_label(r'$\Delta$CloudTrend$\times$100', fontsize=12)
    
    #     Panel labels
    panel_txt = [chr(i) for i in range(ord('a'), ord('r')+1)]
    ax_list = [axbig1, axbig2, ax11, ax12, ax13, ax14, ax21, ax22, ax23, ax24,ax31, ax32, ax33, ax34,ax41, ax42, ax43, ax44]
    for i, a in enumerate(ax_list):
        if i>1:
            plot_subplot_label(a, panel_txt[i], -0.1)
        else:
            plot_subplot_label(a, panel_txt[i])
       
    plt.savefig('../figure/figure_floss.png',dpi=300,bbox_inches='tight')
    plt.savefig('../figure/figure_floss.pdf',bbox_inches='tight')
    print('Figure saved')

if __name__ == '__main__':
    make_plot()
