import reverse_geocoder as rg
import io
import numpy as np
import pandas as pd
import os, netCDF4
import requests

from shapely.geometry import mapping, shape
from shapely.prepared import prep
from shapely.geometry import Point

os.chdir('/Users/hausfath/Desktop/Climate Science/Carbon Brief/Warming Map/')

'''
Create a gridcell metadata csv identifying the name of the closest large city
to the centroid of each gridcell over land.
'''

df = pd.read_csv('simplemaps-worldcities-basic.csv', encoding='utf_8')

df['lat_round'] = (df['lat'] + 0.5).round(0) - 0.5
df['lon_round'] = (df['lng'] + 0.5).round(0) - 0.5

df = df[df['pop'] > 20000]
df.rename(columns = {'country':'country_alt'}, inplace = True)
df.sort_values(by=['lat_round', 'lon_round', 'pop'], ascending=False, inplace=True)
major_cities = df.drop_duplicates(subset=['lat_round', 'lon_round'], keep='first')


data = requests.get("https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson").json()

countries = {}
for feature in data["features"]:
    geom = feature["geometry"]
    country = feature["properties"]["ADMIN"]
    countries[country] = prep(shape(geom))

def get_country(lon, lat):
    point = Point(lon, lat)
    for country, geom in countries.iteritems():
        if geom.contains(point):
            return country
    return "unknown"

lon2d, lat2d = np.meshgrid(np.arange(-179.5, 180.5, 1), np.arange(-89.5, 90.5, 1))
gridded_lat_lons = np.dstack((lat2d, lon2d))

flattened_lat_lons = gridded_lat_lons.reshape(64800,2) #64800 is 180 * 360
coordinates = tuple(map(tuple, flattened_lat_lons)) #[44000:44010]

lats = np.asarray([x[0] for x in coordinates]).astype(float)
lons = np.asarray([x[1] for x in coordinates]).astype(float)

results = rg.search(coordinates)

citynames = []
countrycode = []
country = []
admin1 = []
admin2 = []
geocode_lats = []
geocode_lons = []
full_name = []
place_name = []
country = []
n = 0
for x in results:
    citynames.append(x['name'])
    countrycode.append(x['cc'])
    admin1.append(x['admin1'])
    admin2.append(x['admin2'])
    geocode_lats.append(x['lat'])
    geocode_lons.append(x['lon'])
    full_name.append([''])
    place_name.append([''])
    country.append(get_country(lons[n], lats[n]))
    n += 1

lat_diffs = lats - np.asarray(geocode_lats).astype(float)
lon_diffs = lons - np.asarray(geocode_lons).astype(float)
accurate_coords = (np.absolute(lat_diffs) < 1) * (np.absolute(lon_diffs) < 1)

names = pd.DataFrame({'citynames' : citynames,
                      'countrycode' : countrycode,
                      'admin1' : admin1,
                      'admin2' : admin2,
                      'grid_lats' : lats,
                      'grid_lons' : lons,
                      'country' : country})

#Merge with list of cities over 20k; replace citynames and admin1 with those if in same gridcell
result = pd.merge(names, major_cities, how='left', left_on=['grid_lats', 'grid_lons'], right_on=['lat_round', 'lon_round'])
result['citynames'] = np.where(result['city_ascii'].notnull(), result['city_ascii'], result['citynames'])
result['admin1'] = np.where(result['province'].notnull(), result['province'], result['admin1'])
citynames = result['citynames'].values
admin1 = result['admin1'].values
country = np.where(result['country_alt'].notnull(), result['country_alt'], country)

for n in range(0, len(results)):
    print 'Analyzing grid cell number: ' + str(n)
    if country[n] == 'unknown' and pd.isnull(result['city_ascii'].iloc[n]) == True:
        citynames[n] = 'Lat: '+str(lats[n])+', Long: '+str(lons[n])+', over the ocean'
        countrycode[n] = ''
        admin1[n] = ''
        admin2[n] = ''
        place_name[n] = 'Lat: '+str(lats[n])+', Long: '+str(lons[n])
        country[n] = 'Over the ocean'
        full_name[n] = citynames[n]
    elif accurate_coords[n] == False:
        citynames[n] = 'Lat: '+str(lats[n])+', Long: '+str(lons[n])+', '+country[n]
        countrycode[n] = ''
        admin1[n] = ''
        admin2[n] = ''
        place_name[n] = 'Lat: '+str(lats[n])+', Long: '+str(lons[n])
        full_name[n] = citynames[n]
    elif admin2[n] == '' and accurate_coords[n] == True:
        full_name[n] = 'Near '+citynames[n] + ', ' + admin1[n] + ', ' + country[n]
        place_name[n] = citynames[n] + ', ' + admin1[n]
    elif admin2[n] == '' and admin1[n] == '' and accurate_coords[n] == True:
        full_name[n] = 'Near '+citynames[n]+ ', ' + country[n]
        place_name[n] = citynames[n]
    elif admin1[n] == '' and accurate_coords[n] == True:
        full_name[n] = 'Near '+citynames[n]+ ', ' + country[n]
        place_name[n] = citynames[n]
    else:
        full_name[n] = 'Near '+citynames[n] + ', ' + admin1[n] + ', ' + country[n]
        place_name[n] = citynames[n] + ', ' + admin1[n]


df = pd.DataFrame({'full_name' : full_name,
                   'place_name' : place_name,
                   'citynames' : citynames,
                   'countrycode' : countrycode,
                   'admin1' : admin1,
                   'admin2' : admin2,
                   'grid_lats' : lats,
                   'grid_lons' : lons,
                   'geocode_lats' : np.asarray(geocode_lats).astype(float),
                   'geocode_lons' : np.asarray(geocode_lons).astype(float),
                   'country' : country})

df.to_csv('gridcell_metadata.csv', encoding='utf-8')

unflattened_lat_lons = flattened_lat_lons.reshape((180, 360, 2))