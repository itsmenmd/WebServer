from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import mysql.connector
from datetime import datetime 
from mysql.connector import Error
from nomalize import max_temp_mean, max_temp_std, min_temp_mean, min_temp_std, temp_mean, temp_std, dataset_mean, dataset_std


########################-LOAD DATA-####################################
def predict_all():
    try:
        connection = mysql.connector.connect(host='localhost', database='Weather', user='root', password='22081997')
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
            print("You're connected to database: ", record)
            sql = "select max_temp, min_temp from daily order by id desc limit 70;"
            cursor.execute(sql)
            result = cursor.fetchall()
            max_temp = [] 
            min_temp = []
            for row in result:
                temp1 = []
                temp2 = []          
                temp1.append(row[0])
                temp2.append(row[1])
                max_temp.append(temp1)
                min_temp.append(temp2)
            max_temp.reverse()
            min_temp.reverse()
            sql = "select temp,hump,weather from hourly order by id desc limit 120;"
            cursor.execute(sql)
            result = cursor.fetchall()
            data = []
            for row in result:
                current = []
                for i in range (0,3):
                    current.append(row[i])
                data.append(current)
            data.reverse()
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

    max_temp = np.asarray(max_temp)
    max_temp = [max_temp]
    max_temp = (max_temp - max_temp_mean)/max_temp_std
    min_temp = np.asarray(min_temp)
    min_temp = [min_temp]
    min_temp = (min_temp - min_temp_mean)/min_temp_std

    data = np.asarray(data)
    data = data[:,[0,1]]
    data = [data]
    data = (data - dataset_mean)/dataset_std

    temp_model = tf.keras.models.load_model('temp_model.h5')
    hump_model = tf.keras.models.load_model('hump_model.h5')
    max_model = tf.keras.models.load_model('max_model.h5')
    min_model = tf.keras.models.load_model('min_model.h5')
    ww_model = tf.keras.models.load_model('weather_model.h5')

    predict_temp = temp_model.predict(data)[0]
    predict_hump = hump_model.predict(data)[0]
    data_ww = []
    for i in range (predict_temp.size):
        cur = []
        cur.append(predict_temp[i])
        cur.append(predict_hump[i])
        data_ww.append(cur)
    data_ww = np.asarray(data_ww)

    predict_ww = ww_model.predict(data_ww)
    val_all_ww = ''
    for i in range (0,12):   
        val_all_ww = val_all_ww + str(np.argmax(predict_ww[i])) + ','
    val_all_ww = val_all_ww[:-1]

    predict_temp = predict_temp*temp_std + temp_mean
    val_all_temp = ''
    for i in range(predict_temp.size):
        val_all_temp = val_all_temp + str(int(round(predict_temp[i]))) + ','
    val_all_temp = val_all_temp[:-1]

    predict_max = max_model.predict(max_temp)[0]
    predict_max = predict_max*max_temp_std + max_temp_mean
    val_all_max = ''
    for i in range(predict_max.size):
        val_all_max = val_all_max + str(int(round(predict_max[i]))) + ','
    val_all_max = val_all_max[:-1]

    predict_min = min_model.predict(min_temp)[0]
    predict_min = predict_min*min_temp_std + min_temp_mean
    val_all_min = ''
    for i in range(predict_min.size):
        val_all_min = val_all_min + str(int(round(predict_min[i]))) + ','
    val_all_min = val_all_min[:-1]


    try:
        connection = mysql.connector.connect(host='localhost', database='Weather', user='root', password='22081997')
        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("select database();")
            record = cursor.fetchone()
            print("You're connected to database: ", record)
            sql = "delete from prediction"
            cursor.execute(sql)
            sql = "delete from weathercondition"
            cursor.execute(sql)
            sql = "delete from max_predict"
            cursor.execute(sql)
            sql = "delete from min_predict"
            cursor.execute(sql)
            sql = "insert into prediction (hour1,hour2,hour3,hour4,hour5,hour6,hour7,hour8,hour9,hour10,hour11,hour12) values (%s);" %(val_all_temp)
            cursor.execute(sql)
            sql = "insert into weathercondition (hour1,hour2,hour3,hour4,hour5,hour6,hour7,hour8,hour9,hour10,hour11,hour12) values (%s);" %(val_all_ww)
            cursor.execute(sql)
            sql = "insert into max_predict (day1,day2,day3, day4, day5, day6, day7) values (%s);" %(val_all_max)
            cursor.execute(sql)
            sql = "insert into min_predict (day1,day2,day3, day4, day5, day6, day7) values (%s);" %(val_all_min)
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

predict_all()
while 1:
    time  = datetime.now()
    if(time.minute == 1 and time.second == 0):
        predict_all()