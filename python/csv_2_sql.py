
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np 
import os
import pandas as pd 
import mysql.connector 
from mysql.connector import Error
from common import mysql_conn

data = pd.read_csv("weather_data2.csv", encoding='utf-8')
features = ['Local time in Ha Noi (airport)','T','U','WW']
dataset = data[features]
dataset = dataset.fillna(0)
dataset = dataset.values
datasetTime = dataset[:,0]
datasetTemp  = dataset[:,1]
datasetHump = dataset[:,2]
datasetWeather = dataset[:,3]

try:
    # connection = mysql.connector.connect(host='localhost', database='Weather', user='root', password='root')
    connection=mysql_conn()
    if connection.is_connected():
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)
        sql = "drop table daily;"
        cursor.execute(sql)
        sql = "create table daily (id int(10) primary key auto_increment, Date char(20), max_temp int(10), min_temp int(10));"
        cursor.execute(sql)
        sql = "drop table hourly;"
        cursor.execute(sql)
        sql = "create table hourly (id int(10) primary key auto_increment, Datetime char(20), temp int(10), hump int(10), weather int(10));"
        cursor.execute(sql)
        i = 0
        k=0
        for k in range(len(dataset)):
            sql = "insert into hourly (Datetime, temp, hump, weather) values ('%s',%s,%s,%s);" %(datasetTime[k],datasetTemp[k],datasetHump[k],datasetWeather[k])
            cursor.execute(sql)
            connection.commit()
        while i<len(dataset)-1:
            j=i+1
            date = datasetTime[i][:datasetTime[i].find(" ")]
            min_temp=datasetTemp[i]
            max_temp=datasetTemp[i]
            while datasetTime[j][:datasetTime[j].find(" ")] == date and j<len(dataset)-1:
                if(max_temp<datasetTemp[j]):
                    max_temp=datasetTemp[j]
                if(min_temp>datasetTemp[j]):
                    min_temp=datasetTemp[j]
                j += 1
            i=j
            sql = "insert into daily (date, max_temp, min_temp) values ('%s',%s,%s);" %(date,max_temp,min_temp)
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

    