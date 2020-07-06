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

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

########################-LOAD DATA-####################################

data = pd.read_csv("weather_data.csv", encoding='utf8')
features = ['T', 'U', 'P', 'Ff', 'WW']
dataset = data[features]
dataset.index = data['Local time in Ha Noi (airport)']
dataset = dataset.fillna(0)

########################-NOMARLIZE DATA-##########################

TRAIN_SPLIT = 25000
dataset = dataset.values
datasetnan = dataset[:,4]
temp_mean = dataset[:TRAIN_SPLIT,0].mean(axis=0)
temp_std = dataset[:TRAIN_SPLIT,0].std(axis=0)
dataset_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
dataset_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset - dataset_mean)/dataset_std

###########################-DEVIDE TRAIN DATA-#############################3

def variate_data(dataset, target, start_index, end_index, history_size, target_size, multi=False):
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        if multi:
            data.append(dataset[indices])
        else:
            data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(target[i:i+target_size])
    return np.array(data), np.array(labels)

past_history = 120
future_target = 12

x_train_temp, y_train_temp = variate_data(dataset[:,[0,1]],dataset[:,0], 0, TRAIN_SPLIT, past_history, future_target, multi=True)
x_val_temp , y_val_temp = variate_data(dataset[:,[0,1]],dataset[:,0], TRAIN_SPLIT, None, past_history, future_target, multi=True)

# x_train_ww, y_train_ww = variate_data(dataset[:,[0,1]], datasetnan, 0, TRAIN_SPLIT, 0, 0, multi=True)
# x_val_ww , y_val_ww = variate_data(dataset[:,[0,1]], datasetnan, TRAIN_SPLIT, None, 0, 0, multi=True)

x_train_hump, y_train_hump = variate_data(dataset[:,[0,1]],dataset[:,1], 0, TRAIN_SPLIT, past_history, future_target, multi=True)
x_val_hump , y_val_hump = variate_data(dataset[:,[0,1]],dataset[:,1], TRAIN_SPLIT, None, past_history, future_target, multi=True)

x_train_ww = dataset[:TRAIN_SPLIT,[1,2,3]]
y_train_ww = datasetnan[:TRAIN_SPLIT]
x_val_ww = dataset[TRAIN_SPLIT:len(dataset),[1,2,3]]
y_val_ww = datasetnan[TRAIN_SPLIT:len(dataset)]

###################################-CHANGE TO 3D DATA-################################

BATCH_SIZE = 256
BUFFER_SIZE = 1000

train_data_temp = tf.data.Dataset.from_tensor_slices((x_train_temp, y_train_temp))
train_data_temp = train_data_temp.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_data_temp = tf.data.Dataset.from_tensor_slices((x_val_temp, y_val_temp))
val_data_temp = val_data_temp.batch(BATCH_SIZE).repeat()

temp_model = tf.keras.models.load_model('temp_model.h5')
for x, y in val_data_temp.take(1):
  temp_predict = temp_model.predict(x)[0]*temp_std + temp_mean
  val_all = ''
  for i in range(temp_predict.size):
      val_all = val_all + str(int(round(temp_predict[i]))) + ','
val_all = val_all[:-1]
print(val_all)
try:
    connection = mysql.connector.connect(host='localhost', database='Weather', user='root', password='22081997')
    if connection.is_connected():
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)
        sql = "delete from prediction"
        cursor.execute(sql)
        sql = "insert into prediction (hour1,hour2,hour3,hour4,hour5,hour6,hour7,hour8,hour9,hour10,hour11,hour12) values (%s);" %(val_all)
        cursor.execute(sql)
        connection.commit()
        print("SUCCESSFUL!")
except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    if (connection.is_connected()):
        cursor.close()
        connection.close()
        print("MySQL connection is closed")