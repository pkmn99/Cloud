import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
#from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib as mpl

#att5=xr.open_dataset('../data/results/xu/MODIS_attribu_new.nc')
#msgatt=xr.open_dataset('../data/results/xu/MSG_attribu_new.nc')
att5=xr.open_dataset('../data/results/xu/Modis_attribution0908.nc')
msgatt=xr.open_dataset('../data/results/xu/MSG_attribution0908.nc')


# Plot figure
# MODIS Panel 
discmap5 = mpl.colors.ListedColormap(['red', 'blue','yellow', 'lime','tab:pink'])
#discmap5 = mpl.colors.ListedColormap(['blue','red','lime','yellow','tab:pink'])

v1=np.round((att5.attribution==1).sum()/(att5.attribution>0).sum()*100,0)
v2=np.round((att5.attribution==2).sum()/(att5.attribution>0).sum()*100,0)
v3=np.round((att5.attribution==3).sum()/(att5.attribution>0).sum()*100,0)
v4=np.round((att5.attribution==4).sum()/(att5.attribution>0).sum()*100,0)
v5=np.round((att5.attribution==5).sum()/(att5.attribution>0).sum()*100,0)

fig = plt.figure(figsize=[10,5])
pos1 = [0.05, 0.1, 1, 1] # [left, bottom, width, height]
ax1 = fig.add_axes(pos1, projection=ccrs.PlateCarree())

att5.attribution.where(att5.attribution!=0).plot(cmap=discmap5, ax=ax1, add_colorbar=False, rasterized=True)
ax1.set_extent([-180, 180, -60, 80])
ax1.coastlines()

ax1.set_position([ax1.get_position().x0-0.05, ax1.get_position().y0, ax1.get_position().width , ax1.get_position().height])
cbar1_pos = [ax1.get_position().x1+0.025, ax1.get_position().y0, 0.01, ax1.get_position().height]
cax1 = fig.add_axes(cbar1_pos)
cb1 = mpl.colorbar.ColorbarBase(ax=cax1, cmap=discmap5, norm=Normalize(vmin=1, vmax=5) ,
                                        orientation='vertical', ticks=np.arange(1+4/10, 5.6, 4/5))

cb1.ax.set_yticklabels(['Tree+\n(%d%%)'%v1,'Tree$-$\n(%d%%)'%v2,'Orography+\n(%d%%)'%v3,'Orography$-$\n(%d%%)'%v4,
                        'Others\n(%d%%)'%v5], fontsize=10)
cb1.ax.invert_yaxis()
ax1.set_title('Attribution of potential cloud effect of forest (MODIS)',fontsize=14)


# MSG Panel 
pos2 = [-0.075, 0.22, 0.4, 0.4] # [left, bottom, width, height]
ax2 = fig.add_axes(pos2, projection=ccrs.PlateCarree())
msgatt.attribution.where(msgatt.attribution!=0).plot(cmap=discmap5, ax=ax2, add_colorbar=False, rasterized=True) # tab10, set3
ax2.set_extent([-70, 60, -20, 45]) 
ax2.coastlines()
ax2.set_title('MSG')

#plt.savefig('../figure/figure_attribution.pdf',bbox_inches='tight')
plt.savefig('../figure/figure_attribution_0908.png',dpi=300,bbox_inches='tight')

print('Figure saved')
