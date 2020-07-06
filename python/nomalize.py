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


########################-LOAD DATA-####################################

try:
    connection = mysql.connector.connect(host='localhost', database='Weather', user='root', password='root')
    if connection.is_connected():
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)
        sql = "select max_temp,min_temp from daily;"
        cursor.execute(sql)
        result = cursor.fetchall()
        max_temp = []
        min_temp = []
        for row in result:
            max_temp.append(row[0])
            min_temp.append(row[1])
        max_temp = np.asarray(max_temp)
        min_temp = np.asarray(min_temp)
        sql = "select temp,hump,weather from hourly;"
        cursor.execute(sql)
        result = cursor.fetchall()
        dataset = []
        for row in result:
            current = []
            for i in range(0,3):
                current.append(row[i])
            dataset.append(current)
        dataset = np.asarray(dataset)
except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    if (connection.is_connected()):
        cursor.close()
        connection.close()
        print("MySQL connection is closed")

########################-NOMARLIZE DATA-##########################

TRAIN_SPLIT = 25000
datasetnan = dataset[:,2]
temp_mean = dataset[:TRAIN_SPLIT,0].mean(axis=0)
temp_std = dataset[:TRAIN_SPLIT,0].std(axis=0)
dataset = dataset[:,[0,1]]
dataset_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
dataset_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset - dataset_mean)/dataset_std

TRAIN_SPLIT2 = 1000
max_temp_mean = max_temp[:TRAIN_SPLIT2].mean()
max_temp_std = max_temp[:TRAIN_SPLIT2].std()
min_temp_mean = min_temp[:TRAIN_SPLIT2].mean(axis=0)
min_temp_std = min_temp[:TRAIN_SPLIT2].std(axis=0)
max_temp = (max_temp - max_temp_mean)/max_temp_std
min_temp = (min_temp - min_temp_mean)/min_temp_std