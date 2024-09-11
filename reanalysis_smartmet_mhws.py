#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 11:41:28 2023

author: Veera Haapaniemi (FMI)

"""
import datetime
import pandas as pd 
import requests
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import time
import xarray as xr

from matplotlib import colors
import pickle

import math
import cmocean
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cartopy
import cartopy.io.shapereader as shpreader

varname = 'TW_PT1M_AVG'
current_datetime = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")

def GET_temperature_data_from_SMARTMET(station_id,start_dt,end_dt, savename):
    """
    Function for fetching data from SmartMet for given station name and 
    time period. Returns the data as pandas DataFrame.

    Parameters
    ----------
    station_id : int
    start_dt : datetime
        UTC
        utctime

    end_dt : datetime
        UTC

    Returns
    -------
    df : pd DataFrame
        fetched data
    """
    url = 'http://smartmet.fmi.fi/timeseries'
    payload = {
        "fmisid" : "{}".format(station_id),
        "producer": "observations_fmi",
        "precision": "auto", 			#automatic precision
        "tz":"utc",
        "param": "stationname, stationlat, stationlon," \
            "fmisid," \
            "utctime," \
             'TW_PT1M_AVG',\
        "starttime": "{}".format(start_dt.strftime("%Y-%m-%dT%H:%M:%S")), 
        "endtime": "{}".format(end_dt.strftime("%Y-%m-%dT%H:%M:%S")),
        "timestep": "data",
        "format": "json"
        }

    running = True
    while running:
        try:
            r = requests.get(url, params=payload)
            print(r.url)
            running = False
        except: 
            print("Connection refused by the server (Max retries exceeded)")
            print("Taking a nap...")
            print("ZZzzzz...")
            time.sleep(10)
            print("Slept for 10 secods, now continuing...")
    
    dictr = r.json() 
   
    df = pd.json_normalize(dictr)
    try:
        df['utctime']= pd.to_datetime(df['utctime'])
        df.to_csv('/work/data/haapanie/2024/WT_'+savename+\
                  '_{}.csv'.format(station_id), index=False)
        return df
    except:
        print('empty dataframe')
        df.to_csv('/work/data/haapanie/2024/WT_'+savename+\
                  '_{}.csv'.format(station_id), index=False)
        return  df
    
def plot_helsinki_on_map(coast_lons, coast_lats, y, stations, label, title):
    fig = plt.figure()
    fig.set_size_inches(11,8)

    projection = ccrs.PlateCarree()

    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent([24, 26, 59.6, 60.6])      
    cmap = plt.colormaps['Blues']
    cmap.set_bad(color='white')

    ax.add_feature(cartopy.feature.OCEAN, color='cornflowerblue')
    ax.add_feature(cartopy.feature.LAND, edgecolor='lightblue', color='lightblue')

    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.collections import PatchCollection
    from matplotlib.ticker import FuncFormatter
    from matplotlib import cm, colors

    
    mhw_colors = ['white','gold','darkorange','crimson','darkred']
    cmap = colors.LinearSegmentedColormap.from_list(name='mymap',\
                                                    colors=mhw_colors,N=5)
    cmap._init()
    
    bounds = [0,1,2,3,4]
    norm = colors.Normalize(vmin=0, vmax=4)
    
    names = ['No heatwave','I: Moderate','II: Strong',\
                             'III: Severe','IV: Extreme']
    
 
    # Changing the approach
    upper, lower, N = bounds[0], bounds[-1], len(bounds)   
    # Mapper
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    deltac = (upper - lower)/(2*(N-1))

    mapper.set_array([lower - deltac, upper + deltac])
        
    # Plot the scatter    
    a = plt.scatter(coast_lons, coast_lats, c=y, s=100, marker='o',\
                    transform=projection, edgecolor='k', rasterized=True,\
                    cmap=cmap, norm=norm)

    cbar = plt.colorbar(mapper, ax=plt.gca())
    cbar.set_ticks(bounds)
    cbar.set_ticklabels(names)

    plt.suptitle('Marine heatwave class near Helsinki',fontsize=14)
    plt.title('Last updated '+ title, fontsize=12)

    for i in range(len(stations)):
        sellon = coast_lons[i]
        sellat = coast_lats[i]
        if ((sellon >= 24) & (sellon <=26) & (sellat >= 59.5) & (sellat <= 61)):
            if 'Harmaja' in stations[i]:
                t = plt.text(coast_lons[i]+0.7, coast_lats[i]-0.05, stations[i],\
                         fontsize=12,\
                         horizontalalignment='right',
                         transform=ccrs.Geodetic())
                t.set_bbox(dict(facecolor='white',alpha=0.8))

            elif 'Suomenlinna' in stations[i]:
                t = plt.text(coast_lons[i]+1, coast_lats[i]+0.05, stations[i],\
                             fontsize=12,\
                             horizontalalignment='right',
                             transform=ccrs.Geodetic())
                t.set_bbox(dict(facecolor='white',alpha=0.8))

            elif 'Espoo' in stations[i]:
                t = plt.text(coast_lons[i]-0.1, coast_lats[i], stations[i],\
                             fontsize=12,\
                             horizontalalignment='right',
                             transform=ccrs.Geodetic())
                t.set_bbox(dict(facecolor='white',alpha=0.8))
            elif 'Suomenlah' in stations[i]:
                t = plt.text(coast_lons[i], coast_lats[i]-0.06, stations[i],\
                             fontsize=12,\
                             horizontalalignment='right',
                             transform=ccrs.Geodetic())
                t.set_bbox(dict(facecolor='white',alpha=0.8))
                
    # Define the xticks for longitude
    ax.set_xticks(np.arange(24,26,0.5), crs=projection)
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

    # Define the yticks for latitude
    ax.set_yticks(np.arange(60,61,0.5), crs=projection)
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    plt.savefig('/work/data/haapanie/comparison_plots/reanalysis_nearby_helsinki.png',\
             dpi=600, bbox_inches='tight')
    plt.close()
    return

def plot_on_map(coast_lons, coast_lats, y,  stations, label, title):
    fig = plt.figure()
    fig.set_size_inches(11,9)

    projection = ccrs.PlateCarree()

    ax = plt.axes(projection=ccrs.Mercator())
    ax.set_extent([19, 29, 59, 66])
  
    cmap = plt.colormaps['Blues']
    cmap.set_bad(color='white')

    ax.add_feature(cartopy.feature.OCEAN, color='cornflowerblue')
    ax.add_feature(cartopy.feature.LAND, edgecolor='lightblue', color='lightblue')

    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.collections import PatchCollection
    from matplotlib.ticker import FuncFormatter
    from matplotlib import cm, colors

    
    mhw_colors = ['white','gold','darkorange','crimson','darkred']
    cmap = colors.LinearSegmentedColormap.from_list(name='mymap',\
                                                    colors=mhw_colors,N=5)
    cmap._init()
    
    bounds = [0,1,2,3,4]
    norm = colors.Normalize(vmin=0, vmax=4)
    
    names = ['No heatwave','I: Moderate','II: Strong',\
                             'III: Severe','IV: Extreme']
    
 
    # Changing the approach
    upper, lower, N = bounds[0], bounds[-1], len(bounds)   
    # Mapper
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    deltac = (upper - lower)/(2*(N-1))

    mapper.set_array([lower - deltac, upper + deltac])
    
    
    # Plot the scatter    
    a = plt.scatter(coast_lons, coast_lats, c=y, s=100, marker='o',\
                    transform=projection, edgecolor='k', rasterized=True,\
                    cmap=cmap, norm=norm)

    cbar = plt.colorbar(mapper, ax=plt.gca())
    cbar.set_ticks(bounds)
    cbar.set_ticklabels(names)

    plt.suptitle('Marine heatwave class in the Baltic Sea based on\n'+\
                 'comparison with climatology from a data assimilated\n'+\
                 'reanalysis product covering years 1991-2020',fontsize=10)
    plt.title('Last updated '+title, fontsize=10)
  
    for i in range(len(stations)):
        if 'Helsinki' in stations[i]:
            print('skipping Hki')
        elif 'Suomen' in stations[i]:
            print('skipping Suomen')
        elif 'Espoo' in stations[i]:
            print('skipping Espoo')
        elif 'Selk' in stations[i]:
            t = plt.text(coast_lons[i]+3, coast_lats[i]+0.3, stations[i], fontsize=8,\
                     horizontalalignment='right',
                     transform=ccrs.Geodetic())
            t.set_bbox(dict(facecolor='white',alpha=0.8))
        elif 'Pori' in stations[i]:
            t = plt.text(coast_lons[i]+2.5, coast_lats[i]+0.1, stations[i], fontsize=8,\
                     horizontalalignment='right',
                     transform=ccrs.Geodetic())
            t.set_bbox(dict(facecolor='white',alpha=0.8))
        elif 'Uusika' in stations[i]:
            t = plt.text(coast_lons[i]+4, coast_lats[i]+0.1, stations[i], fontsize=8,\
                     horizontalalignment='right',
                     transform=ccrs.Geodetic())
            t.set_bbox(dict(facecolor='white',alpha=0.8))
        elif 'Pohjois-' in stations[i]:
            t = plt.text(coast_lons[i]+5, coast_lats[i]-0.03, stations[i], fontsize=8,\
                     horizontalalignment='right',
                     transform=ccrs.Geodetic())
            t.set_bbox(dict(facecolor='white',alpha=0.8))
        elif 'Kotka' in stations[i]:
            t = plt.text(coast_lons[i]+1, coast_lats[i]+0.25, stations[i], fontsize=8,\
                     horizontalalignment='right',
                     transform=ccrs.Geodetic())
            t.set_bbox(dict(facecolor='white',alpha=0.8))
        elif 'Hanko' in stations[i]:
            t = plt.text(coast_lons[i]-1, coast_lats[i]-0.1, stations[i], fontsize=8,\
                     horizontalalignment='right',
                     transform=ccrs.Geodetic())
            t.set_bbox(dict(facecolor='white',alpha=0.8))
        elif 'Kemiönsaari' in stations[i]:
            t = plt.text(coast_lons[i]-1, coast_lats[i]+0.1, stations[i], fontsize=8,\
                     horizontalalignment='right',
                     transform=ccrs.Geodetic())
            t.set_bbox(dict(facecolor='white',alpha=0.8))
        else:
            t = plt.text(coast_lons[i]+3, coast_lats[i]+0.2, stations[i], fontsize=8,\
                     horizontalalignment='right',
                     transform=ccrs.Geodetic())
            t.set_bbox(dict(facecolor='white',alpha=0.8))
        t = plt.text(24.5615+0.6, 60.1015+0.15, 'Helsinki', fontsize=8,\
                 horizontalalignment='right',
                 transform=ccrs.Geodetic())
        t.set_bbox(dict(facecolor='white',alpha=0.8))
    
    # Define the xticks for longitude
    ax.set_xticks(np.arange(19,29,3), crs=projection)
    lon_formatter = cticker.LongitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)

    # Define the yticks for latitude
    ax.set_yticks(np.arange(59,66,3), crs=projection)
    lat_formatter = cticker.LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    plt.savefig('/work/data/haapanie/comparison_plots/reanalysis_all_stations.png',\
             dpi=600, bbox_inches='tight')
    plt.close()
    return

def plot_no_mhw(df_climate, t, maxs, p90, means, p10, mins, df_avg_year, varname):
    """
    

    Parameters
    ----------
    df_climate : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    maxs : TYPE
        DESCRIPTION.
    p90 : TYPE
        DESCRIPTION.
    means : TYPE
        DESCRIPTION.
    p10 : TYPE
        DESCRIPTION.
    mins : TYPE
        DESCRIPTION.
    df_avg_year : TYPE
        DESCRIPTION.
    varname : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # -- year 2024 compared to the rest of the data
    fig, axs = plt.subplots(1, 1, sharex=True, figsize=(14, 8))

    plt.suptitle(df_climate.stationname[0], fontsize=16)
    axs.set_title('Time series last updated '+ current_datetime, fontsize=12)
    
    # Graphs
    axs.plot(t,maxs,label='Max',color='tab:blue',alpha=0.3)
    axs.plot(t,p90,label='90 %',color='tab:blue',alpha=0.6)
    axs.plot(t,means,label='Mean',color='tab:blue')
    axs.plot(t,p10,label='10 %',color='tab:blue',alpha=0.6)
    axs.plot(t,mins,label='Min',color='tab:blue',alpha=0.3)
    #
    # Adding some color
    axs.fill_between(t, mins, maxs, alpha=0.2,color='tab:blue')
    axs.fill_between(t, p10, p90, alpha=0.4,color='tab:blue')  
    # Another plot
    axs.plot(df_avg_year['2024-01-01':'2024-12-31'][varname].index,\
             df_avg_year['2024-01-01':'2024-12-31'][varname],\
             color='tab:red',lw=3,label='Year 2024')   
    # Formatting dates
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axs.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
      
    # Background
    major_ticks = np.arange(0, 30, 5)
    minor_ticks = np.arange(0, 30, 1)
    axs.set_yticks(major_ticks)
    axs.set_yticks(minor_ticks, minor=True)
    # Or if you want different settings for the grids:
    axs.grid(which='minor', alpha=0.3, axis='y', ls='--')
    axs.grid(which='major', alpha=0.7, axis='y',ls='-')
    axs.grid(axis='x',ls=':',alpha=0.7)
    
    # Limits
    axs.set_xlim(t[0],t[-1])
    axs.set_ylim(-1,30)
    # Labels etc
    axs.set_ylabel('Temperature')
    plt.legend()
    #plt.show()
    plt.savefig('/work/data/haapanie/comparison_plots/reanalysis_Temperatures_{}.png'.format(ID),\
                        dpi=600, bbox_inches='tight')
    plt.close()
    return

def mhw_reanalysis(clim, df_2024):
    clim = clim.to_dataframe().reset_index().set_index('time')
    # Seven day means used currently for the climatology
    clim = clim.resample('7D').mean(numeric_only=True)
    clim = clim.resample('d').interpolate(method='linear') 
    clim = clim.to_xarray()
        
    clim['doy'] = clim.time.dt.dayofyear
    print(clim.doy)
    # Compute the thresholds
    
    # Transform climatology to reference year
    clim_dates = pd.to_datetime(clim.doy-2, unit='D', origin=str(2024))
        
# -----------------------------------------------------------
    print('-------------------here 1.0')
    print(clim_dates)

    x1_threshold = pd.DataFrame(data={'date': clim_dates, varname: clim.pctl_90})
    print(x1_threshold)
    means = pd.DataFrame(data={'date':clim_dates, varname: clim.timmean})
        
    x2 = (x1_threshold - means)*2 + means
    x3 = (x1_threshold - means)*3 + means
    x4 = (x1_threshold - means)*4 + means
    x5 = (x1_threshold - means)*5 + means
           
    x2_threshold = pd.DataFrame(data={'date': clim_dates, varname: x2[varname]})
    x3_threshold = pd.DataFrame(data={'date': clim_dates, varname: x3[varname]})
    x4_threshold = pd.DataFrame(data={'date': clim_dates, varname: x4[varname]})
    x5_threshold = pd.DataFrame(data={'date': clim_dates, varname: x5[varname]})
        
    x1_threshold = x1_threshold.set_index('date')
    x2_threshold = x2_threshold.set_index('date')
    x3_threshold = x3_threshold.set_index('date')
    x4_threshold = x4_threshold.set_index('date')
    x5_threshold = x5_threshold.set_index('date')
        
    if df_2024.empty:
        print('no data yet!')
    else:
        df_2024.index = pd.to_datetime(df_2024['utctime'])
        df_2024 = df_2024[df_2024[varname] != -99]  
        df_avg_year = df_2024.resample('D').mean(numeric_only=True)

        print('here')
        print(df_avg_year)
            
        t_start = df_avg_year.index[0]
        t_end = df_avg_year.index[-1]
            
        threshold = x1_threshold[(x1_threshold.index >= t_start) &\
                                 (x1_threshold.index <= t_end)]
            
        print(df_avg_year)
        print(threshold)
        # Selecting the MHW events
        print('Temperatures over 90 percentile')
        year_mhws = df_avg_year[(df_avg_year[varname] > threshold[varname])]
        year_mhws['class_days_above_90'] = year_mhws.index.to_series()\
                                            .diff().dt.days.ne(1).cumsum()
            
        # Creating an index for each block of days above 90th percentile        
        count_days_above_90 = year_mhws['class_days_above_90'].value_counts()
        list_mhw_events = count_days_above_90[count_days_above_90 > 5].index.values
            
        print('--- MHW events listed ----')
        print(list_mhw_events)
            
        for element in list_mhw_events:
            print(element)
            select_mhw_event = year_mhws[year_mhws.class_days_above_90 == element]
            print(select_mhw_event)
                
        # Comparing to todays measurements
        today = datetime.datetime.today()
        date = today.strftime('%Y-%m-%d')     
            
        try:               
            latest = year_mhws[year_mhws.class_days_above_90 == max(list_mhw_events)] 
            latest_dates = latest.index.strftime('%Y-%m-%d').values
                
            mhw_start = latest_dates[0]
            mhw_end = latest_dates[-1]
                
            # Setting the thresholds for todays temperatures
            class_2_today = x2_threshold[x2_threshold.index.strftime('%Y-%m-%d')\
                                     == date].values
            class_3_today = x3_threshold[x3_threshold.index.strftime('%Y-%m-%d')\
                                     == date].values
            class_4_today = x4_threshold[x4_threshold.index.strftime('%Y-%m-%d')\
                                     == date].values

            if date not in latest_dates:
                print('No marine heatwave! Setting class to 0')
                mhw_index = 0
            if date in latest_dates:
                print('Marine heatwave ongoing, setting the class tentatively to one')
                mhw_index = 1
                
            print('Checking the classification')
            temperature_today = year_mhws[year_mhws.index.strftime('%Y-%m-%d') \
                                      == date][varname].values
            print(temperature_today)
                
            if temperature_today > class_2_today:
                print('Strong heatwave ongoing')
                mhw_index = 2
            if temperature_today > class_3_today:
                print('Severe heatwave ongoing')
                mhw_index = 3
            if temperature_today > class_4_today:
                print('Extreme heatwave ongoing')
                mhw_index = 4            
        except IndexError:
            mhw_index = 0
        except UnboundLocalError:
            print('what')
            mhw_index=0
        except ValueError:
            mhw_index = 0
        
    t = clim_dates
    maxs = clim.timmax
    p90 = clim.pctl_90
    means = clim.timmean
    p10 = clim.pctl_10
    mins = clim.timmin
    x2 =  x2[varname]
    x3 = x3[varname]
    x4 = x4[varname]
    x5 = x5[varname]
    
    if mhw_index == 0:
        print('no heatwave')
        plot_no_mhw(df_2024, t, maxs, p90, means,\
                    p10, mins, df_avg_year, varname)
            
    else: 
        # --- heatwave ongoing, adding the subplot
        # -- year 2024 compared to the rest of the data
        fig, axs = plt.subplots(2, 1, sharex=False, figsize=(14, 12))
            
        # ------- Subplot 1
        # Title
        plt.suptitle(df_2024.stationname[0], fontsize=16)
        axs[0].set_title('Time series last updated '+ current_datetime, fontsize=12)
        
        # Graphs                        
        axs[0].plot(t,maxs,label='Max',color='tab:blue',alpha=0.3)
        axs[0].plot(t,p90,label='90 %',color='tab:blue',alpha=0.6)
        axs[0].plot(t,means,label='Mean',color='k', alpha=0.5)
        axs[0].plot(t,p10,label='10 %',color='tab:blue',alpha=0.6)
        axs[0].plot(t,mins,label='Min',color='tab:blue',alpha=0.3)
        # Adding some color
        axs[0].fill_between(t, mins, maxs, alpha=0.2,color='tab:blue')
        axs[0].fill_between(t, p10, p90, alpha=0.4,color='tab:blue')   
        # Another plot
        axs[0].plot(df_avg_year['2024-01-01':'2024-12-31'][varname].index,\
                    df_avg_year['2024-01-01':'2024-12-31'][varname],\
                    color='tab:red',lw=5,label='Year 2024')        
        # Formatting dates
        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        axs[0].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        # Background
        major_ticks = np.arange(0, 30, 5)
        minor_ticks = np.arange(0, 30, 1)
        axs[0].set_yticks(major_ticks)
        axs[0].set_yticks(minor_ticks, minor=True)
        # Or if you want different settings for the grids:
        axs[0].grid(which='minor', alpha=0.3, axis='y', ls='--')
        axs[0].grid(which='major', alpha=0.7, axis='y',ls='-')
        axs[0].grid(axis='x',ls=':',alpha=0.7)


        # Limits
        axs[0].set_xlim(t[0],t[-1])
        axs[0].set_ylim(-1,30)
        
        # ------- Subplot 2
        mhw_start = latest_dates[0]
        mhw_end = latest_dates[-1]
        
        c_list = ['white','gold','darkorange','crimson','darkred']
        axs[1].set_title('Marine Heatwave Class')
        axs[1].plot(t,means,label='Mean',color='k',alpha=0.5)
        axs[1].fill_between(t, p90, x2, alpha=0.2, color=c_list[1])
        axs[1].fill_between(t, x2, x3, alpha=0.2, color=c_list[2])
        axs[1].fill_between(t, x3, x4, alpha=0.2, color=c_list[3])
        axs[1].fill_between(t, x4, x5, alpha=0.2, color=c_list[4])

        axs[1].plot(t, p90, label='I: Temperature over 90th percentile (p90)', color=c_list[1],alpha=0.3)
        axs[1].plot(t, x2, label='II: Temperature over 2 x p90 difference', color=c_list[2],alpha=0.3)
        axs[1].plot(t, x3, label='III: Temperature over 3 x p90 difference',color=c_list[3],alpha=0.3)
        axs[1].plot(t, x4, label='IV: Temperature over 4 x p90 difference',color=c_list[4],alpha=0.3)
      
        # Another plot
        axs[1].plot(df_avg_year['2024-01-01':'2024-12-31'][varname].index,\
                    df_avg_year['2024-01-01':'2024-12-31'][varname],\
                     color='tab:red',lw=5,label='Year 2024')        
        # +/- 15 days
        start_comparison = today - datetime.timedelta(days = 15)
        end_comparison = today + datetime.timedelta(days = 10)
        axs[1].set_xlim([start_comparison, end_comparison])
        axs[1].set_ylim(-1,30)

        
        # Background
        major_ticks = np.arange(0, 30, 5)
        minor_ticks = np.arange(0, 30, 1)
        axs[1].set_yticks(major_ticks)
        axs[1].set_yticks(minor_ticks, minor=True)
        # Or if you want different settings for the grids:
        axs[1].grid(which='minor', alpha=0.3, axis='y', ls='--')
        axs[1].grid(which='major', alpha=0.7, axis='y',ls='-')
        axs[1].grid(axis='x',ls=':',alpha=0.7)
        
        axs[1].legend()
        # Labels etc
        axs[0].set_ylabel('Temperature')
        axs[1].set_ylabel('Temperature')
        axs[0].legend()
        plt.savefig('/work/data/haapanie/comparison_plots/reanalysis_Temperatures_{}.png'.format(ID),\
                        dpi=600, bbox_inches='tight')
       # plt.show()
        plt.close()      
    return mhw_index

if __name__ == '__main__':

    all_ids = ['103807', '137228', '103808', '107033', '134246',\
               '104600','106631','654923',\
               '654900', '654910',\
               '103976','100996', '134221', '100761', '134220']
    
    station_lons = []
    station_lats = []
    current_classes = []
    mhw_indices = []
    station_names = []
    
    for ID in all_ids:
        print('considering ID = ', ID)
        
        # Read climatology from CMEMS reanalysis
        clim = xr.open_dataset('/work/data/haapanie/climatology/full_climatology_'\
                               +ID+'.nc')
        clim = clim.rename({'mean':'timmean', 'min':'timmin', 'max':'timmax'})
        
        # Fetch temperature data for 2024            
        start_dt = datetime.datetime(2024,1,1,1)
        end_dt = datetime.datetime(2024,12,31,23)
        savename = '2024'
        
        print('Fetching 2024 data')
        GET_temperature_data_from_SMARTMET(ID,start_dt,end_dt, savename)
        print('2024 ok')
        
        try:
            df_2024 = pd.read_csv('/work/data/haapanie/2024/WT_'+\
                               '2024_{}.csv'.format(ID))
            print('2024 dataset reading ok')
            mhw_idx = mhw_reanalysis(clim, df_2024)
            
            station_lats.append(np.nanmean(df_2024.stationlat))
            station_lons.append(np.nanmean(df_2024.stationlon))
            station_names.append(df_2024.stationname[0])
            mhw_indices.append(mhw_idx)
     
        except pd.errors.EmptyDataError:
            print('no data to fetch')

    current_datetime = datetime.datetime.now().strftime("%d.%m.%Y %H:%M")
            
    plot_on_map(station_lons, station_lats, mhw_indices, station_names,\
                'Marine Heatwave Class', current_datetime)
    plot_helsinki_on_map(station_lons, station_lats, mhw_indices, station_names,\
                         'Marine Heatwave Class', current_datetime)
