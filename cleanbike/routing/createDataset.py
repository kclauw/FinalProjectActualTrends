import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from shapely import geometry
import seaborn as sns
import datetime
import random
import pyproj

now = datetime.datetime.now()

try:
    import pycpt
    cmap = pycpt.load.cmap_from_cptcity_url('ukmo/wow/temp-c.cpt')
except:
    cmap = 'Spectral_r'


column_names = 'numbername;address;position;banking;bonus;status;contract_name;bike_stand;available_bike_stand;available_bikes;last_update'.split(";")


df = pd.read_csv('./data/villo.csv', names=column_names,sep=";")

#Split position column into geocoordinates


df['latitude'], df['longitude'] = df['position'].str.split(',', 1).str
df = df[['latitude','longitude','address','position']]



#Generate random data




mu, sigma = 70, 10 # mean and standard deviation
s = np.random.normal(mu, sigma, len(df['latitude']))
df_final = pd.DataFrame(data=None, columns=df.columns,index=df.index)
days=1
mu, sigma = 70, 10 # mean and standard deviation
all_res = []
for i in range(0,days):
    df2 = df
    #Random number between a certain range
    r = random.randrange(-10, 10)
    s = np.random.normal(mu+r, sigma, len(df['latitude']))
    df2['pm2.5'] = s
    df_final.append(df2)
    all_res.append(df2)
    print df



df = pd.concat(all_res)

df.latitude = df.latitude.astype(float).fillna(0.0)
df.longitude = df.longitude.astype(float).fillna(0.0)
# North American Datum 1927
p1 = pyproj.Proj(proj='latlong', datum='NAD27')
# WGS84 Latlong
p2 = pyproj.Proj(proj='latlong', datum='WGS84')
# WGS84 UTM Zone 16
p3 = pyproj.Proj(proj='utm', zone=16, datum='WGS84')
df['long_wgs84'], df['lat_wgs84'] = pyproj.transform(p1, p2, 
                                                            df.longitude.values, 
                                                            df.latitude.values)
df['E_utm'], df['N_utm'] = pyproj.transform(p1, p3,
                                                        df.longitude.values,
                                                        df.latitude.values)
df['geometry'] = [geometry.Point(x, y) for x, y in zip(df['long_wgs84'], df['lat_wgs84'])]

df.to_csv('./data/processed_data.csv')