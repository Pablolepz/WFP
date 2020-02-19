import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlite3
import os
import googlemaps


gmaps = googlemaps.Client('AIzaSyB0Jxv9PEsH6uRd4PS4c_1L3mKzy95thic')

# Data import =================================================================

curr_path = os.path.dirname(__file__)
data_path = os.path.relpath('../../data_WPF', curr_path)
query = "SELECT FOD_ID,LATITUDE,LONGITUDE,STAT_CAUSE_DESCR,FIRE_YEAR FROM Fires WHERE STATE LIKE '%CA%' and FIRE_YEAR > 2010;"

print(data_path + "/FPA_FOD_20170508.sqlite")
conn = sqlite3.connect("D:\\Documents\\College\\Spring2020\\Projects\\COMP542\\data_WPF\\FPA_FOD_20170508.sqlite")
# con = sqlite3.connect('database.db')
# cursor = con.cursor()
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# print(cursor.fetchall())
# # conn.row_factory = sqlite3.Row
# print(conn)
df = pd.read_sql_query(query, conn)
# df = pd.read_sql(conn)
print(df)



# Bar chart plot ===========================================================
# countList = []
# print(df['STAT_CAUSE_DESCR'].value_counts())
# countListVals = dict(df['STAT_CAUSE_DESCR'].value_counts()).values()
# countListNames = dict(df['STAT_CAUSE_DESCR'].value_counts()).keys()
# plt.figure(figsize=(9,3))
#
# plt.subplot()
# plt.bar(countListNames,countListVals)
# plt.suptitle('Events')
# plt.axis([0,24,0,50])
# plt.xticks(rotation=90)
# # for label in ax.get_xaxis().get_ticklabels()[::2]:
# #     label.set_visible(False)
# # plt.show()
#
#
# # Google maps generation ====================================================
col_one_list = df['LATITUDE'].tolist()
col_two_list = df['LONGITUDE'].tolist()
