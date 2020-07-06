from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import mysql.connector 
from mysql.connector import Error
# from test_day import max_temp_mean, max_temp_std

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

########################-LOAD DATA-####################################

data = pd.read_csv("weather_data2.csv", encoding='utf8')
features = ['T', 'U', 'P', 'Ff', 'WW']
dataset = data[features]
dataset.index = data['Local time in Ha Noi (airport)']
dataset = dataset.fillna(3)

########################-NOMARLIZE DATA-##########################
print(dataset.head())
TRAIN_SPLIT = 25000
dataset = dataset.values
datasetnan = dataset[:,4]
temp_mean = dataset[:TRAIN_SPLIT,0].mean(axis=0)
temp_std = dataset[:TRAIN_SPLIT,0].std(axis=0)
dataset_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
dataset_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset - dataset_mean)/dataset_std

# try:
#     connection = mysql.connector.connect(host='localhost', database='Weather', user='root', password='22081997')
#     if connection.is_connected():
#         cursor = connection.cursor()
#         cursor.execute("select database();")
#         record = cursor.fetchone()
#         print("You're connected to database: ", record)
#         sql = "select max_temp from test3 order by id desc limit 70;"
#         cursor.execute(sql)
#         result = cursor.fetchall()
#         temp1 = []
#         temp2 = []
#         for row in result:          
#             temp1.append(row[0])
#         temp1.reverse()
#         temp2.append(temp1)
# except Error as e:
#     print("Error while connecting to MySQL", e)
# finally:
#     if (connection.is_connected()):
#         cursor.close()
#         connection.close()
#         print("MySQL connection is closed")

# temp2 = np.asarray(temp2)

# temp2 = temp2.reshape(70,1)
# temp2 = [temp2]
# temp2 = (temp2 - max_temp_mean)/max_temp_std

# max_model = tf.keras.models.load_model('max_model.h5')
# predict = max_model.predict(temp2)[0]
# predict = predict*max_temp_std + max_temp_mean
# print(predict)