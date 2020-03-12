import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
import os
# import googlemaps
import sys
# from mpl_toolkits.basemap import Basemap
import georaster as gr
import gdal
import xarray as xr
from tqdm import tqdm
# import rasterio
# gmaps = googlemaps.Client('AIzaSyB0Jxv9PEsH6uRd4PS4c_1L3mKzy95thic')
from rastertodataframe import raster_to_dataframe
from math import sin, cos, sqrt, atan2, radians
from multiprocessing import Pool

#utility functions
def getPercent(num, status):
    if ((num >= len(precDF)/5) and status == 0):
        status = 1
        print("20% complete")
    elif ((num >= len(precDF)/4) and status == 1):
        status = 2
        print("25% complete")
    elif ((num >= (len(precDF)/5) * 2) and status == 2):
        status = 3
        print("40% complete")
    elif ((num >= len(precDF)/2) and status == 3):
        status = 4
        print("50% complete")
    elif ((num >= (len(precDF)/5) * 3) and status == 4):
        status = 5
        print("60% complete")
    elif ((num >= (len(precDF)/4) * 3) and status == 5):
        status = 6
        print("75% complete")
    elif ((num >= (len(precDF)/5) * 4) and status == 6):
        status = 7
        print("80% complete")
    return status



# Data import =================================================================

# https://www1.ncdc.noaa.gov/pub/data/cdo/documentation/PRECIP_HLY_documentation.pdf

#https://www.ncdc.noaa.gov/cdo-web/search using: Precipitation, 2010-2012, 2012 - 2014, counties, california



#======================================================
#============Loading CSV files=========================
#======================================================
print('========== Collecting data... =============')

curr_path = os.path.dirname(__file__)
data_path = os.path.relpath('../../data_WPF', curr_path)
query = "SELECT FOD_ID,LATITUDE,LONGITUDE,STAT_CAUSE_DESCR,FIRE_YEAR FROM Fires WHERE STATE LIKE '%CA%' and FIRE_YEAR >= 2010;"

print(data_path + "/FPA_FOD_20170508.sqlite")
conn = sqlite3.connect("D:\\\Documents\\College\\Spring2020\\Projects\\COMP542\\data_WPF\\FPA_FOD_20170508.sqlite")

df = pd.read_sql_query(query, conn)

print(df)


#precipitation data ============

df1 = pd.read_csv("D:\\Documents\\College\\Spring2020\\Projects\\COMP542\\data_WPF\\Precip_data\\2010-2012.csv")

df2 = pd.read_csv("D:\\Documents\\College\\Spring2020\\Projects\\COMP542\\data_WPF\\Precip_data\\2012-2014.csv")

#biome data =============
biomeDS = xr.open_rasterio("D:\\Documents\\College\\Spring2020\\Projects\\COMP542\\data_WPF\\biomes\\anthromes_v2_2000\\a2000_na.tif")


# driver = gdal.GetDriverByName('GTiff')
# filename = "D:\\Documents\\College\\Spring2020\\Projects\\COMP542\\data_WPF\\biomes\\anthromes_v2_2000\\a2000_no.tif" #path to raster
# dataset = gdal.Open(filename)
# band = dataset.GetRasterBand(1)
#
# cols = dataset.RasterXSize
# rows = dataset.RasterYSize
#
# transform = dataset.GetGeoTransform()
#
# xOrigin = transform[0]
# yOrigin = transform[3]
# pixelWidth = transform[1]
# pixelHeight = -transform[5]
#
# data = band.ReadAsArray(0, 0, cols, rows)
#
# points_list = [ (355278.165927, 4473095.13829), (355978.319525, 4472871.11636) ] #list of X,Y coordinates
#
# for point in points_list:
#     col = int((point[0] - xOrigin) / pixelWidth)
#     row = int((yOrigin - point[1] ) / pixelHeight)
#
#     print row,col, data[row][col]
# biomeCSV = pd.read_csv("D:\\Documents\\College\\Spring2020\\Projects\\COMP542\\data_WPF\\biomes\\anthromes_v2_2000\\Oa2000_na.csv")

#========================

print(df1)
print(df2)
print(biomeDS)





print('========== data preprocessing... =======')
# print(biomeDS.head(1))
# print(biomeDS.tail(1))

#======================================================
#=============Filtering CSVs to LLAs===================
#=============and required data========================

# Precipitation LLA/HPCP/DATE=========

# HPCP is rain in inches
print('filtering precipitation data...')
df1 = df1[['STATION', 'STATION_NAME', 'LATITUDE', 'LONGITUDE', 'HPCP', 'DATE']]
df2 = df2[['STATION', 'STATION_NAME','LATITUDE', 'LONGITUDE', 'HPCP', 'DATE']]

# print(df1)
# print(df2)

#combine rain data
precDF = df1.append(df2, ignore_index = True)
precDF.reset_index()

#remove rows with 999 HPCP
precDF = precDF[precDF['HPCP'] < 900]

pS = 0

print("selecting only fireseason data")
#remove percipitaton during the fire season
fireOffSeaStart = 501
fireOffSeaEnd = 1131
#create a new data set of precipitation in average HPCP during California fire off season (based on last year's season: May - November)
precDF['DATE'] = precDF['DATE'].str[:8]
precDF['YEAR'] = precDF['DATE'].str[:4]
precDF = precDF[((precDF['DATE'].astype(int) - precDF['DATE'].str[:4].astype(int) * 10000) > fireOffSeaStart) & ((precDF['DATE'].astype(int) - precDF['DATE'].str[:4].astype(int) * 10000) < fireOffSeaEnd)]



print("averaging rain per day...")
precDF['HPCP'] = precDF.groupby('DATE')['HPCP'].transform('mean')
# print(precDF.head(20))


#set average HPCP for each station
# precDF = precDF.drop_duplicates(subset='DATE', keep = "last").reset_index()
print("averaging rain per station per fireseason")
precDF['HPCP'] = precDF.groupby(['STATION' ,'YEAR'])['HPCP'].transform('mean')
precDF = precDF.reset_index(drop = True)
print(precDF.head(20))

precDF = precDF.drop_duplicates(subset=['STATION', 'YEAR'], keep = "last").reset_index(drop = True)

del precDF['DATE']
precDF['STATION'] = precDF['STATION'].str[6:].astype(int)
precDF = precDF.sort_values(['STATION','YEAR']).reset_index(drop = True)

# for i, l in precDF.iterrows():
#     if i > 1000:
#         break
#     # elif i < 900:
#     #     continue
#     print(i)
#     print(precDF.at[i, 'STATION_NAME'])
#     print(precDF.at[i, 'YEAR'])
#     print(precDF.at[i, 'HPCP'])
# precDF = precDF[precDF['Date']]


#for each station a specific Lat,Lon
# for index, row in precDF.head(n = 2).iterrows():
#     print(index, row)
#     print('current LatLon = '
#     if currStation != row['STATION']:
#         precDF.replace(to_replace = )

print(precDF)

#Sqeeze data to one row per station, with the HPCP data of the fire year



print('========== Wildfire data cleansing... =========')
#Wildfire causation counting

countList = []
print(df['STAT_CAUSE_DESCR'].value_counts())
countListVals = dict(df['STAT_CAUSE_DESCR'].value_counts()).values()
countListNames = dict(df['STAT_CAUSE_DESCR'].value_counts()).keys()

df = df[df.STAT_CAUSE_DESCR != 'Missing/Undefined']
df = df[df.STAT_CAUSE_DESCR != 'Miscellaneous']
df = df[df.FIRE_YEAR <= 2014]

countList = []
print(df['STAT_CAUSE_DESCR'].value_counts())
countListVals = dict(df['STAT_CAUSE_DESCR'].value_counts()).values()
countListNames = dict(df['STAT_CAUSE_DESCR'].value_counts()).keys()

df = df[['LATITUDE', 'LONGITUDE', 'STAT_CAUSE_DESCR','FIRE_YEAR']]
df.reset_index()
print(df)


#======================================================
#=============Biome data processing====================
#======================================================
print('========== Biome data preprocessing... =========')
# print(ds.head)
# def biome_numbers(argument):
#     switch = {
#         11: "Urban",
#         12: "Mixed settlements",
#         21: "Rice villages",
#         22: "Irrigated villages",
#         23: "Rainfed villages",
#         24: "Pastoral villages",
#         31: "Residential irrigated croplands",
#         32: "Residential rainfed croplands",
#         33: "Populated croplands",
#         34: "Remote croplands",
#         41: "Residential rangelands",
#         42: "Populated rangelands",
#         43: "Remote rangelands",
#         51: "Residential woodlands",
#         52: "Populated woodlands",
#         53: "Remote woodlands",
#         54: "Inhabited treeless and barren lands",
#         61: "Wild woodlands",
#         62: "Wild treeless and barren lands"
#     }
#     return switch.get(argument, "NoData")
biomeList = []
pS = 0
#debug purpose try
# try:
for i, c in precDF.iterrows():
    # if i > 1000:
    #     break
    # elif i < 900:
    #     continue
    # print(i, c)
    pS = getPercent(i, pS)
    try:
        pixVal = biomeDS.sel(band=0, y=precDF.loc[i, 'LATITUDE'], x = precDF.loc[i, 'LONGITUDE'], method='nearest').values
        # if (type(pixVal) is not np.ndarray):
            # print("=============")
            # print(precDF.at[i, 'STATION_NAME'])
            # print(precDF.at[i, 'LATITUDE'])
            # print("...")
            # print(precDF.at[i, 'LONGITUDE'])
            # print("-----")
            # print(i)
            # print("&&&&&&")
            # print(pixVal)
            # print("=============")
        # else:
            # print("nonononosnaofdscnbosahvbocywneag oiu2ewvgFOCIAUEW")
        biomeList.append(pixVal)
    except:
        print("Biome Unknown")
        biomeList.append(999)

# except:
    # print("Error. Old pixVal at {}, i at {}".format(pixVal,i))
    # try:
    #     y=precDF.loc[i, 'LATITUDE']
    #     print("1latitude worked")
    # except:
    #     try:
    #         x=precDF.loc[i, 'LONGITUDE']
    #         print("1.1longitude worked")
    #     except:
    #         print("1.2longitude failed")
    # try:
    #     x=precDF.loc[i, 'LONGITUDE']
    #     print("2longitude worked")
    # except:
    #     try:
    #         y=precDF.loc[i, 'LATITUDE']
    #         print("2.1latitude worked")
    #     except:
    #         print("2.2latitude failed")
    # raise Exception("Code dun work")
    # print("For Station {}, biome is: {}".format(precDF.loc[i, 'STATION'],pixVal))
print("length of biome list {}".format(len(precDF)))
print("length of biome list {}".format(len(biomeList)))

precDF['STAT_BIOME'] = biomeList

# def ifBiome(x):
#     if (isinstance(x, list)):
#         print("biome edge")
#         if (max(list) > 999):
#             return true


#drop unknown biomes
precDF = precDF[precDF.STAT_BIOME != 999]
print(precDF.head(5))



#======================================================
#============= Fire Checker Data ======================
#======================================================
FireBinary = []
#
#station perimeter in km
stat_perim = 5
#
#
# approximate radius of earth in km
R = 6373.0
def get_fire_distance(stat_la, stat_lo, fire_la, fire_lo):
    # print("getting fire from: {},{},{},{},".format(stat_la, stat_lo, fire_la, fire_lo))
    # print(type(fire_lo))
    # print(fire_lo)
    # print(type(stat_lo))
    # print(stat_lo)
    dlon = radians(fire_lo - stat_lo)
    dlat = radians(fire_la - stat_la)

    a = sin(dlat / 2)**2 + cos(radians(stat_la)) * cos(radians(fire_la)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance
def get_fire_res(stat_la, stat_lo, YEAR):
    #iterate through firedataset and find a fire in the vicinity
    for w, u in df.iterrows():
        fDist = get_fire_distance(stat_la,stat_lo,df.loc[w,'LATITUDE'].astype(float),df.loc[w,'LONGITUDE'].astype(float))
        # print(fDist)
        if (fDist < stat_perim):
            if (YEAR == df.loc[w,'FIRE_YEAR']):
                return True
            else:
                return False
    print("whaaaaaaaaaat")
    return False


tqdm.pandas()
precDF['FIRE'] = precDF.progress_apply(lambda x: get_fire_res(float(x['LATITUDE']), float(x['LONGITUDE']), int(x['YEAR'])), axis= 1)

# print(precDF.head(20))
# for i, l in precDF.iterrows():
#     if i > 200:
#         break
#     elif i < 100:
#         continue
#     # print(i)
#     # if (precDF.at[i, 'FIRE?'] == True):
#         # print('======================================================')
#         # print(i)
#         # print(precDF.at[i, 'STATION_NAME'])
#     print(precDF.at[i, 'YEAR'])
#     print(type(precDF.at[i, 'YEAR']))
#         # print(precDF.at[i, 'HPCP'])
#     print(precDF.at[i, 'FIRE?'])
#======================================================
#=============Master Dataset Creation==================
#======================================================
print('========== Master dataframe creation... =========')


#create a new data set of precipitation in average HPCP during California fire off season (based on last year's season: May - November)

print('==bebop==')
# print(get_fire_distance(37.312552, -120.851447, 39.276098, -123.094567))
print(precDF.FIRE.value_counts())

precDF.to_csv (r'D:\Documents\College\Spring2020\Projects\COMP542\data_WPF\masterDF.csv', index = False, header = True)
print('==bebop2==')

#======================================================
#=============Training Model===========================
#======================================================

#======================================================
#=============Testing Model============================
#======================================================

#======================================================
#=============OLD Biome data processing================
#======================================================

#
# print('===================================')
# print('===================================')
# raster = 'D:\\Documents\\College\\Spring2020\\Projects\\COMP542\\data_WPF\\biomes\\a1900_na.tif'
# #
# # data = gr.from_file(raster)
# #
# # data.plot()
#
# ds = gdal.Open(raster)
# if ds is None:
#     print('Unable to open %s' % input_file)
#     sys.exit(1)
# #     try:
# #         srcband = ds.GetRasterBand(band_num)
# #     except RuntimeError, e:
# #         print 'No band %i found' % band_num
# #         print e
#         # sys.exit(1)
# print('File list:', ds.GetFileList())
# print('Width:', ds.RasterXSize)
# print('Height:', ds.RasterYSize)
# print('Coordinate system:', ds.GetProjection())
# print(ds.GetProjection())
# print('===================================')
# srcband = ds.GetRasterBand(1)
# # print(srcband.NoDataValue())
# print(srcband)
# print('===============read biome data as numpy array=========')
# myarray = np.array(ds.GetRasterBand(1).ReadAsArray())
# print(myarray.shape)
# print(myarray.size)
# print(myarray)
#
# myarray
#
# ds = None
# vector_path = '/some/ogr/compatible/file.geojson'
#======================================================
#============Old Code==================================
#======================================================
#
# # Google maps generation ====================================================
# col_one_list = df['LATITUDE'].tolist()
# col_two_list = df['LONGITUDE'].tolist()
#
# fig = plt.figure(figsize=(8, 8))
# m = Basemap(projection='lcc', resolution='h',
#             lat_0=37.5, lon_0=-119,
#             width=1E6, height=1.2E6)
# m.shadedrelief()
# m.drawcoastlines(color='gray')
# m.drawcountries(color='gray')
# m.drawstates(color='gray')
# #
# # # 2. scatter city data, with color reflecting population
# # # and size reflecting area
# m.scatter(col_one_list, col_two_list, latlon=True,
#           c=np.log10(population), s=area,
#           cmap='Reds', alpha=0.5)

# 3. create colorbar and legend
# plt.colorbar(label=r'$\log_{10}({\rm population})$')
# plt.clim(3, 7)

# make legend with dummy points
# for a in [100, 300, 500]:
#     plt.scatter([], [], c='k', alpha=0.5, s=a,
#                 label=str(a) + ' km$^2$')
# plt.legend(scatterpoints=1, frameon=False,
#            labelspacing=1, loc='lower left');

# markers = []
#
# for point in col_one_list:
    # markers.append(gmaps.)


# print(col_one_list, col_two_list)
#
# geocode_result = gmaps.
# //=======================================



# Extract all image pixels (no vector).
print('===================================')
print('====================BiomeCSV=============')





# # Extract only pixels the vector touches and include the vector metadata.
# df = raster_to_dataframe(raster_path, vector_path=vector_path)
