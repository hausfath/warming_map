import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, netCDF4
from mpl_toolkits.basemap import Basemap, addcyclic, shiftgrid, interp, maskoceans
from statsmodels.nonparametric.smoothers_lowess import lowess


os.chdir('/Users/hausfath/Desktop/Climate Science/Carbon Brief/Warming Map/')

obs_filename = '/Users/hausfath/Desktop/Climate Science/Carbon Brief/Warming Map/Land_and_Ocean_LatLong1_Annual_Uncertainty.nc'
anom_year_start = 1951
anom_year_end = 1980


def import_climate_model_fields(filename, verbose = False):
    '''
    Import climate model fields from netCDF files.
    '''
    nc = netCDF4.Dataset(filename, 'r')
    if verbose == True:
        nc_attrs, nc_dims, nc_vars = ncdump(nc)
    lats = nc.variables['lat'][:]
    lons = nc.variables['lon'][:]
    temps = nc.variables['tas'][:,:,:]
    times = np.arange(len(nc.variables['time']))
    years = 1861 + times
    return {
        'lats' : lats,
        'lons' : lons,
        'temps' : temps,
        'years' : years
    }


def import_berkeley(filename, verbose = False):
    '''
    Import observational temperature fields from Berkeley Earth netCDF file.
    '''
    nc = netCDF4.Dataset(filename, 'r')
    if verbose == True:
        nc_attrs, nc_dims, nc_vars = ncdump(nc)
    lats = nc.variables['latitude'][:]
    lons = nc.variables['longitude'][:]
    temps = nc.variables['temperature'][:,:,:]
    unc = nc.variables['uncertainty'][:,:,:]
    times = nc.variables['time'][:]
    years = times.astype(int)
    return {
        'lats' : lats,
        'lons' : lons,
        'temps' : temps,
        'unc' : unc,
        'times' : times,
        'years' : years
    }


def calc_gridded_anomalies(temps, years, anom_year_start, anom_year_end):
    '''
    Calculate anomalies for each grid cell for each year with respect to
    the provided anomaly start and end years.
    '''
    start_pos = np.where(years == anom_year_start)[0][0]
    end_pos = np.where(years == anom_year_end)[0][0] + 1
    mean = np.nanmean(temps[start_pos:end_pos], axis=0)
    anoms = temps - mean
    return anoms


def downscale_time_series(data, lats, lons, resolution = 1):
    '''
    Downscale timeseries to a 1x1 lat/lon resolution for each timestep.
    Use bilinear interpolation as the default, but use nearest neighbor
    infilling for any gridcells adjacent to areas of missing data.
    '''
    if lons.max() > 200:
        data, lons = shiftgrid(180, data, lons, start=False)
    lons_fine = np.arange(-179.5, 180, resolution)
    lats_fine = np.arange(-89.5, 90, resolution)
    lons_sub, lats_sub = np.meshgrid(lons_fine, lats_fine)
    month = []
    for i in range(data.shape[0]):
        fine_bilinear = interp(
            data[i], lons, lats, lons_sub, lats_sub, checkbounds=False, masked=False, order=1
        )
        fine_nn = interp(
            data[i], lons, lats, lons_sub, lats_sub, checkbounds=False, masked=False, order=0
        )
        fine_bilinear = np.ma.masked_invalid(fine_bilinear)
        mask = np.ma.getmask(fine_bilinear)
        fine = np.ma.where(mask, fine_nn, fine_bilinear)
        try:
            results = np.dstack((results, fine))
        except:
            results = fine
        month.append(i)
    results = np.ma.masked_invalid(results)
    return {
        'anoms' : results,
        'month' : month,
        'lons' : lons_fine,
        'lats' : lats_fine
        }


def get_downscaled_model_data(rcp):
    '''
    Import and downscale climate model data.
    Return climate model data after 1999.
    '''
    model_filename = '/Users/hausfath/Desktop/Climate Science/Carbon Brief/Warming Map/CMIP5_mmm_'+rcp+'.nc'
    model_data = import_climate_model_fields(model_filename)
    model_data['anoms'] = calc_gridded_anomalies(model_data['temps'], model_data['years'], anom_year_start, anom_year_end)
    model_data['years'] = model_data['years'][138:]
    downscalled = downscale_time_series(model_data['anoms'][138:], model_data['lats'], model_data['lons'], 1)
    return {
        'years' : model_data['years'],
        'model_data' : downscalled['anoms'],
        }
    return downscalled


def rebaseline_model_to_obs(model_data, obs_data, model_years, obs_years, baseline_start, baseline_end):
    '''
    Rebaseline climate model data to match observations during the 1999-2018 period.
    '''
    model_data = np.swapaxes(model_data,0,2)
    model_data = np.swapaxes(model_data,1,2)
    start_pos = np.where(model_years == baseline_start)[0][0]
    end_pos = np.where(model_years == baseline_end)[0][0] + 1
    model_mean = np.nanmean(model_data[start_pos:end_pos], axis=0)

    start_pos = np.where(obs_years == baseline_start)[0][0]
    end_pos = np.where(obs_years == baseline_end)[0][0] + 1
    obs_mean = np.nanmean(obs_data[start_pos:end_pos], axis=0)
    rebased_model = model_data - model_mean + obs_mean
    rebased_model = np.swapaxes(rebased_model,0,2)
    rebased_model = np.swapaxes(rebased_model,0,1)
    return rebased_model




def csv_each_gridcell(obs_data, annual_obs, rcp85, rebased_26, rebased_45, rebased_60, rebased_85):
    '''
    Produce a csv file for each gridcell containing observations, lowess-smoothed observations,
    and the four RCP scenarios for each 1x1 lat/lon gridcell on the globe. Add a header line
    that includes the header metadata.
    '''
    n=0
    for lat in range(0,180):
        for lon in range(0,360):
            obs = pd.DataFrame({'year' : obs_data['years'],
                               'obs_anoms' : annual_obs[lat,lon].round(2),
                               'uncertainty' : obs_data['unc'][lat,lon].round(2)})
            model = pd.DataFrame({'year' : rcp85['years'],
                                  'rcp26' : rebased_26[lat,lon].round(2),
                                  'rcp45' : rebased_45[lat,lon].round(2),
                                  'rcp60' : rebased_60[lat,lon].round(2),
                                  'rcp85' : rebased_85[lat,lon].round(2)})

            result = pd.merge(obs, model, how='outer', on=['year'])
            first_valid = result['obs_anoms'].first_valid_index()
            last_valid_obs = result['obs_anoms'].last_valid_index()
            bwidth = 10. / (last_valid_obs - first_valid)
            smoothed_data = lowess(result['obs_anoms'], result['year'], is_sorted=True, frac=bwidth)
            smooth = pd.DataFrame({'year' : smoothed_data[:,0],
                                   'smoothed_anoms' : smoothed_data[:,1].round(2)})
            result = pd.merge(result, smooth, how='outer', on=['year'])
            result = result[['year', 'obs_anoms', 'smoothed_anoms', 'uncertainty', 'rcp26', 'rcp45', 'rcp60', 'rcp85']]
            header_lines = gridcell_metadata['full_name'][n]
            lat_label = obs_data['lats'][lat]
            lon_label = obs_data['lons'][lon]
            print 'Saving gridcell '+str(n)+' of 64800'

            os.chdir('/Users/hausfath/Desktop/Climate Science/Carbon Brief/Warming Map/csvs/')
            result.to_csv('gridcell_'+str(lat_label)+'_'+str(lon_label)+'.csv',header=True,index=True,index_label=header_lines, encoding='utf-8')
            n+=1


def gridcell_characteristics_csv(obs_data, annual_obs, rcp85, rebased_26, rebased_45, rebased_60, rebased_85):
    '''
    Produce a singe csv file including lat, lon, location information, observed warming between the past
    ten years and the first thirty years of record at the location, and projected warming between the past
    10 years and 2100 in climate models.
    '''
    n=0
    place_name = []
    country = []
    lat_label = []
    lon_label = []
    obs_warming = []
    model_26_warming = []
    model_45_warming = []
    model_60_warming = []
    model_85_warming = []

    for lat in range(0,180):
        for lon in range(0,360):
            obs = pd.DataFrame({'year' : obs_data['years'],
                               'obs_anoms' : annual_obs[lat,lon].round(2)})
            model = pd.DataFrame({'year' : rcp85['years'],
                                  'rcp26' : rebased_26[lat,lon].round(2),
                                  'rcp45' : rebased_45[lat,lon].round(2),
                                  'rcp60' : rebased_60[lat,lon].round(2),
                                  'rcp85' : rebased_85[lat,lon].round(2)})

            result = pd.merge(obs, model, how='outer', on=['year'])
            result = result[['year', 'obs_anoms', 'rcp26', 'rcp45', 'rcp60', 'rcp85']]

            place_name.append(gridcell_metadata['place_name'][n])
            country.append(gridcell_metadata['country'][n])
            lat_label.append(obs_data['lats'][lat])
            lon_label.append(obs_data['lons'][lon])

            first_valid = result['obs_anoms'].first_valid_index()
            last_valid_obs = result['obs_anoms'].last_valid_index()

            first_30_years = result['obs_anoms'][first_valid:first_valid+30].mean()

            obs_warming.append(round(result['obs_anoms'][last_valid_obs - 10:last_valid_obs].mean() - first_30_years, 2))
            model_26_warming.append(round(result['rcp26'][-10:].mean(), 2))
            model_45_warming.append(round(result['rcp45'][-10:].mean(), 2))
            model_60_warming.append(round(result['rcp60'][-10:].mean(), 2))
            model_85_warming.append(round(result['rcp85'][-10:].mean(), 2))
            print 'Calculating gridcell '+str(n)+' of 64800'
            n+=1

    metadata_csv = pd.DataFrame({'place_name' : place_name,
                                 'country' : country,
                                 'lat_label' : lat_label,
                                 'lon_label' : lon_label,
                                 'obs_warming' : obs_warming,
                                 'model_26_warming' : model_26_warming,
                                 'model_45_warming' : model_45_warming,
                                 'model_60_warming' : model_60_warming,
                                 'model_85_warming' : model_85_warming})

    print metadata_csv.head()
    metadata_csv.to_csv('gridcell_characteristics.csv', encoding='utf-8')


gridcell_metadata = pd.read_csv('gridcell_metadata.csv', encoding='utf-8')

#Process obs data
obs_data = import_berkeley(obs_filename)
annual_obs = calc_gridded_anomalies(obs_data['temps'], obs_data['years'], anom_year_start, anom_year_end)

#Process model data
rcp26 = get_downscaled_model_data('26')
rebased_26 = rebaseline_model_to_obs(rcp26['model_data'], annual_obs, rcp26['years'], obs_data['years'], 1999, 2017)

rcp45 = get_downscaled_model_data('45')
rebased_45 = rebaseline_model_to_obs(rcp45['model_data'], annual_obs, rcp45['years'], obs_data['years'], 1999, 2017)

rcp60 = get_downscaled_model_data('60')
rebased_60 = rebaseline_model_to_obs(rcp60['model_data'], annual_obs, rcp60['years'], obs_data['years'], 1999, 2017)

rcp85 = get_downscaled_model_data('85')
rebased_85 = rebaseline_model_to_obs(rcp85['model_data'], annual_obs, rcp85['years'], obs_data['years'], 1999, 2017)

#Reshape obs data to have time dimension last
annual_obs = np.moveaxis(annual_obs,0,-1)
obs_data['unc'] = np.moveaxis(obs_data['unc'],0,-1)

#Generate csvs for each gridcell
csv_each_gridcell(obs_data, annual_obs, rcp85, rebased_26, rebased_45, rebased_60, rebased_85)

#Generate gridcell characteristics csv
gridcell_characteristics_csv(obs_data, annual_obs, rcp85, rebased_26, rebased_45, rebased_60, rebased_85)

