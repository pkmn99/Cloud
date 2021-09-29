import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import numpy as np
import matplotlib as mpl
from scipy import stats
import statsmodels.stats.api as sms
from matplotlib.colors import ListedColormap, Normalize
#from matplotlib.colors import Normalize
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# Revised figure 1 by merging attribution part with the potential effect 

def get_myjet_cmap(levels=False,left_offset=0,right_offset=0,cmap='myjet'):
    if cmap=='myjet':
        mycmap=np.genfromtxt('../data/jet_blue_red_colormap.csv', delimiter=',')
    else:
        mycmap = plt.get_cmap(cmap)(range(256))
    if levels.any():
        n1=np.floor(np.linspace(127, 0, num=np.int(levels.shape[0]/2)))
        n2=np.floor(np.linspace(128, 255, num=np.int(levels.shape[0]/2)))
        n=np.concatenate([n1[::-1],n2]).astype(int)
        if right_offset!=0:
            n=np.concatenate([n1[::-1],n2[0:-right_offset]]).astype(int)
        if left_offset!=0:
            n=np.concatenate([n1[::-1],n2[left_offset::]]).astype(int)
        return ListedColormap(mycmap[n])
    else:
        return ListedColormap(mycmap)

# Reproduce latitude statistics from matlab, data should be numpy data array
def lat_pattern(data):
    lat_width = 1 # default is 1 degree  
    dim=data.shape #[3600,7200] # row, coloum
    res=180/dim[0]
    lat_number = int(lat_width/res) # number of 0.05 pixels to be aggregated
    lat=np.zeros([6,int(dim[0]/lat_number)])
    lat[0,:]=np.arange(-90+lat_width/2,90+lat_width/2,lat_width)

    k=0;
    for i in range(0,int(dim[0]-lat_number),lat_number):
        temp=data[i:i+int(lat_number),:].flatten()
        temp=temp[~np.isnan(temp)]

        lat[1,k]=temp.shape[0] # sample number
        lat[2,k]=np.mean(temp) # difference
        if stats.ttest_1samp(temp,0).pvalue<0.05:
            lat[3,k]=1  # 1: significant; -1 not significant
        else: 
            lat[3,k]=-1
        lat[4,k],lat[5,k]=sms.CompareMeans(sms.DescrStatsW(temp), # lower and upper CI
                                           sms.DescrStatsW(np.zeros(temp.shape))).tconfint_diff(usevar='unequal')
        k=k+1;
    return lat

# Add lat lon to map figure
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

def make_plot():
    # Load data
    ds05=xr.open_dataset('../data/results/xu/MODIS_potential_0207.nc')
    msghour=xr.open_dataset('../data/results/msg_potential_maximum_hour0908.nc')
    msg14=xr.open_dataset('../data/results/xu/MSG_potential0908.nc')
    att5=xr.open_dataset('../data/results/xu/MODIS_attribution_0207.nc')
    msgatt=xr.open_dataset('../data/results/xu/MSG_attribution0908.nc')
    
    # Calculate some variables
    # latitudinal statistics for cloud enhancment and inhibition
    pot_pos_lat=lat_pattern(ds05.potential.where(ds05.potential>0).values)
    pot_neg_lat=lat_pattern(ds05.potential.where(ds05.potential<0).values)
    pot_pos_lat_msg=lat_pattern(msg14.potential.where(msg14.potential>0).values)
    pot_neg_lat_msg=lat_pattern(msg14.potential.where(msg14.potential<0).values)

    # range of bin includes left and exclude right, and include rightmost edge
    f1=np.histogram(msghour.local_hour.values.flatten(), bins=range(0,25), density=True) 
    v1=np.round(f1[0][0:6].sum()*100,0)
    v2=np.round(f1[0][6:12].sum()*100,0)
    v3=np.round(f1[0][12:18].sum()*100,0)
    v4=np.round(f1[0][18:24].sum()*100,0)
    # percent of positive and negative cloud effects
    pos_modis = (ds05.potential>0).sum()/(ds05.potential.notnull()).sum()*100
    neg_modis = (ds05.potential<0).sum()/(ds05.potential.notnull()).sum()*100
    pos_msg = (msg14.potential>0).sum()/(msg14.potential.notnull()).sum()*100
    neg_msg = (msg14.potential<0).sum()/(msg14.potential.notnull()).sum()*100
    print('percent for modis %d, for msg %d'%(pos_modis, pos_msg))

    # jet cmap
   # mycmap=get_myjet_cmap(levels=np.arange(-0.15,0.151,0.0125))
   # mycmap_msg=get_myjet_cmap(levels=np.arange(-0.15,0.151,0.0125),left_offset=2)
    mycmap=get_myjet_cmap(levels=np.arange(-0.15,0.151,0.025)) # use levels for color
    mycmap_msg=mycmap #get_myjet_cmap()
    
    fig = plt.figure(figsize=(10,10.5))
    ####################### Panel A
    pos1 = [0.05, 0.65, 0.70, 0.35] # [left, bottom, width, height]
    ax1 = fig.add_axes(pos1, projection=ccrs.PlateCarree())
    ds05.potential.plot(ax=ax1, vmax=0.15,vmin=-0.15, transform=ccrs.PlateCarree(), 
                           cmap=mycmap,add_colorbar=False, rasterized=True)
    ax1.set_ylabel('')
    ax1.set_xlabel('')
    ax1.set_extent([-180, 180, -60, 80])
    set_lat_lon(ax1, range(-120,180,60), range(-60,80,30), label=True,pad=0.05, fontsize=10)
    ax1.coastlines()
    ax1.add_feature(cfeature.LAKES,edgecolor='black',facecolor='None') # Add lake
    
    ################# Panel B            
    pos2 = [ax1.get_position().x1, ax1.get_position().y0, 0.15, ax1.get_position().height]
    ax2 = fig.add_axes(pos2)
    # all  impact
#    ax2.plot(pot_lat[2,:],pot_lat[0,:],lw=1.25,color='k',label='Potential')
#    ax2.fill_betweenx(pot_lat[0,:], pot_lat[4,:], pot_lat[5,:], facecolor='#D2D2D2',edgecolor='grey', alpha=0.8, label='Total')
    # positive impact
    ax2.plot(pot_pos_lat[2,:],pot_pos_lat[0,:],lw=1,color='red',label='Positive')
    ax2.fill_betweenx(pot_pos_lat[0,:], pot_pos_lat[4,:], pot_pos_lat[5,:], facecolor='#D2AAAA',edgecolor='none', alpha=0.8, label='Positive')
    # Negative impact
    ax2.plot(pot_neg_lat[2,:],pot_neg_lat[0,:],lw=1,color='blue',label='Negative')
    ax2.fill_betweenx(pot_neg_lat[0,:], pot_neg_lat[4,:], pot_neg_lat[5,:], facecolor='#D2D2D2',edgecolor='none', alpha=0.8, label='Negative')

    # Add positive/negative cloud effect percentage number
    ax2.text(-0.05, -47, '%d%%'%neg_modis,ha='center',color='blue',fontsize=9)
    ax2.text(0.05, -47, '%d%%'%pos_modis,ha='center',color='red',fontsize=9)

    # Legend
    h1, l1 = ax2.get_legend_handles_labels()
    ax2.legend(h1[0:2],['Positive','Negative'],loc='upper right',fontsize='small',frameon=False,
                       handlelength=1,handletextpad=0.25)
    
    ax2.plot([0,0],[-60,80],'--',lw=1,color='black') # zero line
    ax2.set_ylim([-60,80])
    ax2.set_xlim([-0.075,0.075])
    ax2.set_xlabel('$\Delta$Cloud')
    ax2.set_xticks(np.arange(-0.05, 0.10-0.01, 0.05))
    ax2.set_yticks(np.arange(-60, 80, 30))
    ax2.yaxis.set_ticks_position('both')
    ax2.xaxis.set_ticks_position('both')
    ax2.set_yticklabels(['60S', '30S', '0', '30N', '60N']) 
    ax2.tick_params(labelright=True,labelleft=False)
    ax2.tick_params(axis="x",direction='in', top=True, right=True)
    
    #############Panel C  MSG  potential impact
    pos3 = [0.05, 0.225, 0.3, 0.5] # [left, bottom, width, height] #
    ax3 = fig.add_axes(pos3, projection=ccrs.PlateCarree())
    
    msg14.potential.plot(vmin=-0.15, vmax=0.15,cmap=mycmap_msg,
                            transform=ccrs.PlateCarree(),add_colorbar=False,rasterized=True)
    
    ax3.set_extent([-70, 60, -20, 45])
    ax3.coastlines()
    set_lat_lon(ax3, range(-60,61,60), range(-30,61,30), label=True,pad=0.05, fontsize=10)

    ### Panel D MSG latitude pattern
    pos3b = [ax3.get_position().x1, ax3.get_position().y0, 0.15, ax3.get_position().height]
    ax3b= fig.add_axes(pos3b)

    # positive impact
    ax3b.plot(pot_pos_lat_msg[2,:],pot_pos_lat_msg[0,:],lw=1,color='red',label='Positive')
    ax3b.fill_betweenx(pot_pos_lat_msg[0,:], pot_pos_lat_msg[4,:], pot_pos_lat_msg[5,:], facecolor='#D2AAAA',edgecolor='none', alpha=0.8, label='Positive')
    # Negative impact
    ax3b.plot(pot_neg_lat_msg[2,:],pot_neg_lat_msg[0,:],lw=1,color='blue',label='Negative')
    ax3b.fill_betweenx(pot_neg_lat_msg[0,:], pot_neg_lat_msg[4,:], pot_neg_lat_msg[5,:], facecolor='#D2D2D2',edgecolor='none', alpha=0.8, label='Negative')

    # Add positive/negative cloud effect percentage
    ax3b.text(-0.05, -30, '%d%%'%neg_msg,ha='center',color='blue',fontsize=9)
    ax3b.text(0.05, -30, '%d%%'%pos_msg,ha='center',color='red',fontsize=9)

    # Legend
    h1, l1 = ax3b.get_legend_handles_labels()
    ax3b.legend(h1[0:3],['Positive','Negative'],loc=[0.5,0.5],fontsize='small',frameon=False,
                               handlelength=1,handletextpad=0.25)

    ax3b.plot([0,0],[-30,60],'--',lw=1,color='black') # zero line
    ax3b.set_ylim(ax3.get_ylim())
    ax3b.set_xlim([-0.075,0.075])
    ax3b.set_xlabel('$\Delta$Cloud')
    ax3b.set_xticks(np.arange(-0.05, 0.10-0.01, 0.05))
    ax3b.set_yticks(np.arange(-30, 61, 30))
    ax3b.yaxis.set_ticks_position('both')
    ax3b.xaxis.set_ticks_position('both')
   # ax3b.set_yticklabels(['-30°S', '0°',  '30°N', '60°N']) 
    ax3b.set_yticklabels(['-30S', '0',  '30N', '60N']) 
    ax3b.tick_params(labelright=True,labelleft=False)
    ax3b.tick_params(axis="x",direction='in', top=True, right=True)
    
    ###################Panel E MSG max impact hours 
    pos4 = [0.6, 0.225, 0.3, 0.5] # [left, bottom, width, height]
    ax4 = fig.add_axes(pos4, projection=ccrs.PlateCarree())
    #mycmap2=ListedColormap(['yellow', 'lime','orangered','blue'])
    mycmap2=ListedColormap(['yellow', 'lime','red','blue'])
    
    msghour.local_hour.plot(levels=[0, 6, 12, 18, 24], cmap=mycmap2,
                            transform=ccrs.PlateCarree(),add_colorbar=False,rasterized=True)
    
    ax4.set_extent([-70, 60, -20, 45]) # the range -20 ~ 45 is smaller than the actual display range
    ax4.coastlines()
    set_lat_lon(ax4, range(-60,61,60), range(-30,61,30), label=True,pad=0.05, fontsize=10)
    
    ax4_bar = ax4.inset_axes([0.3,0.1,0.3,0.25])
    
    ##########Panel D inset bar chart
    ax4_bar.bar(f1[1][0:-1]+0.5,f1[0]*100,width=0.75, color=['yellow']*6 + ['lime']*6+['orangered']*6+['blue']*6)
    ax4_bar.text(2.5, 5, '%d'%v1+'%',ha='center',color='yellow',fontsize=9)
    ax4_bar.text(8.5, 8,'%d'%v2+'%',ha='center',color='lime',fontsize=9)
    ax4_bar.text(14.5, 12.7,'%d'%v3+'%',ha='center',color='orangered',fontsize=9)
    ax4_bar.text(20.5, 5, '%d'%v4+'%',ha='center',color='blue',fontsize=9)
    ax4_bar.set_ylim([0,13.5])
    ax4_bar.set_xlim([-0.5,24.5])
    ax4_bar.set_xticks(range(0,25,6))
    ax4_bar.set_xticklabels(range(0,25,6), fontsize=8)
    ax4_bar.tick_params(axis='x', which='major', pad=0)
    ax4_bar.tick_params(axis='y', which='major', pad=0,left=False,labelleft=False)
    ax4_bar.spines['top'].set_visible(False)
    ax4_bar.spines['right'].set_visible(False)
    ax4_bar.spines['left'].set_visible(False)

    ################ Panel F attribution MODIS
    discmap5 = mpl.colors.ListedColormap(['red', 'blue','yellow', 'lime','tab:pink'])
    v1=np.round((att5.attribution==1).sum()/(att5.attribution>0).sum()*100,0)
    v2=np.round((att5.attribution==2).sum()/(att5.attribution>0).sum()*100,0)
    v3=np.round((att5.attribution==3).sum()/(att5.attribution>0).sum()*100,0)
    v4=np.round((att5.attribution==4).sum()/(att5.attribution>0).sum()*100,0)
    v5=np.round((att5.attribution==5).sum()/(att5.attribution>0).sum()*100,0)
    pos5 = [0.05, -0.01, 0.7, 0.35] # [left, bottom, width, height]
    ax5 = fig.add_axes(pos5, projection=ccrs.PlateCarree())
    att5.attribution.where(att5.attribution!=0).plot(cmap=discmap5, ax=ax5, add_colorbar=False, rasterized=True)
    ax5.set_extent([-180, 180, -60, 80])
    ax5.coastlines()
    ax5.text(0.45, 0.05, 'MODIS', fontsize=12,transform=ax5.transAxes,ha='center',fontweight='bold')

    ################ Panel F attribution MSG
    pos6 = [0.05, 0.02, 0.17, 0.17] # [left, bottom, width, height]
    ax6 = fig.add_axes(pos6, projection=ccrs.PlateCarree())
    msgatt.attribution.where(msgatt.attribution!=0).plot(cmap=discmap5, ax=ax6, add_colorbar=False, rasterized=True) # tab10, set3
    ax6.set_extent([-70, 60, -20, 45])
    ax6.coastlines()
    ax6.text(0.25, 0.05, 'MSG', fontsize=12,transform=ax6.transAxes,fontweight='bold')
   # ax6.set_title('MSG')
    
    # Add colorbar
    # Colorbar for panel A
    cbar1_pos = [ax1.get_position().x0+ax1.get_position().width*0.15, ax1.get_position().y0-0.04,  ax1.get_position().width*0.7, 0.01]
    cax1 = fig.add_axes(cbar1_pos)
    cb1 = mpl.colorbar.ColorbarBase(ax=cax1, cmap=mycmap, norm=Normalize(vmin=-0.15, vmax=0.15),
                                    orientation='horizontal', ticks=np.arange(-0.15, 0.16, 0.05)) #cmap=plt.get_cmap('hot')
    cb1.set_label('$\Delta$Cloud', fontsize=12)
    cax1.text(-0.1,0.5,'Less clouds\n over forests', transform=cax1.transAxes, color='b',ha='center',va='center',fontsize=10)
    cax1.text(1.1,0.5,'More clouds\n over forests',transform=cax1.transAxes, color='r',ha='center',va='center',fontsize=10)
    
#    # Colorbar for panel C
#    cbar3_pos = [ax3.get_position().x1+0.025, ax3.get_position().y0, 0.01, ax3.get_position().height]
#    cax3 = fig.add_axes(cbar3_pos)
#    cb3 = mpl.colorbar.ColorbarBase(ax=cax3, cmap=mycmap, norm=Normalize(vmin=-0.15, vmax=0.15),
#                                                                 orientation='vertical', ticks=np.arange(-0.15, 0.16, 0.05)) #cmap=plt.get_cmap('hot')
#    cb3.set_label('$\Delta$Cloud', fontsize=12,labelpad=0)
    
    # Colorbar for panel E
    cbar4_pos = [ax4.get_position().x1+0.025, ax4.get_position().y0, 0.01, ax4.get_position().height]
    cax4 = fig.add_axes(cbar4_pos)
    cb4 = mpl.colorbar.ColorbarBase(ax=cax4, cmap=mycmap2, norm=Normalize(vmin=0, vmax=24),
                                    orientation='vertical', ticks=np.arange(0, 25, 6)) #cmap=plt.get_cmap('hot') 
    cb4.set_label('Local time', fontsize=12)

    # Colorbar for panel F
    cbar5_pos = [ax5.get_position().x1+0.025, ax5.get_position().y0, 0.01, ax5.get_position().height]
    cax5 = fig.add_axes(cbar5_pos)
    cb5 = mpl.colorbar.ColorbarBase(ax=cax5, cmap=discmap5, norm=Normalize(vmin=1, vmax=5) ,
                                                    orientation='vertical', ticks=np.arange(1+4/10, 5.6, 4/5))
    cb5.ax.set_yticklabels(['Tree+\n(%d%%)'%v1,'Tree$-$\n(%d%%)'%v2,'Orography+\n(%d%%)'%v3,'Orography$-$\n(%d%%)'%v4,
                                'Others\n(%d%%)'%v5], fontsize=10)
    cb5.ax.invert_yaxis()
    
    # Add panel label 
    ax1.text(-0.04, 1.01, 'a', fontsize=14, transform=ax1.transAxes, fontweight='bold')
    ax2.text(0, 1.01, 'b', fontsize=14, transform=ax2.transAxes, fontweight='bold')
    ax3.text(-0.09, 1.03, 'c', fontsize=14, transform=ax3.transAxes, fontweight='bold')
    ax3b.text(0, 1.03, 'd', fontsize=14, transform=ax3b.transAxes, fontweight='bold')
    ax4.text(-0.09, 1.03, 'e', fontsize=14, transform=ax4.transAxes, fontweight='bold')
    ax5.text(-0.04, 1.01, 'f', fontsize=14, transform=ax5.transAxes, fontweight='bold')
    
    # Add subplot title
    ax1.set_title('Potential cloud effect ($\Delta$Cloud)')
    ax1.text(0.125, 0.05, 'MODIS', fontsize=12,transform=ax1.transAxes,ha='center',fontweight='bold')
    ax3.set_title('Potential cloud effect ($\Delta$Cloud)')
    ax3.text(0.25, 0.05, 'MSG', fontsize=12,transform=ax3.transAxes,fontweight='bold')
    ax4.set_title('Local hour of maximum effect')
    ax5.set_title('Attribution of potential cloud effect of forests')
    
    plt.savefig('../figure/figure1.pdf', bbox_inches='tight')
#    plt.savefig('../figure/figure1_0730.png', dpi=300, bbox_inches='tight')
    print('Figure saved')

if __name__ == '__main__':
    make_plot()
