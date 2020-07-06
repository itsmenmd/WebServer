import json
import mysql.connector 
from mysql.connector import Error
from datetime import datetime
max_temp = 0
min_temp = 100

def storeDB(jsonData):
    global max_temp
    global min_temp
    json_dict = json.loads(jsonData)
    temp = json_dict['temp']
    hump = json_dict['hum']
    weather = json_dict['ww']
    if(max_temp<temp): max_temp = temp
    if(min_temp>temp): min_temp = temp
    time = datetime.now()
    timeDaily = str(time.day) + '.' +  str(time.month) + '.' + str(time.year)
    timeHourly = timeDaily + " " + str(time.hour) + ":00"
    try:
        connection = mysql.connector.connect(host='localhost', database='Weather', user='root', password='root')
        if connection.is_connected():
            cursor = connection.cursor()
            sql = "delete from datanow"
            cursor.execute(sql)
            sql = "insert into datanow (temp, hump, ww) values (%s,%s,%s);" % (temp,hump,weather)
            cursor.execute(sql)
            connection.commit()
            print("SUCCESSFUL!")         
    except Error as e:
        print("Error while connecting to MySQL", e)
    finally:
        if (connection.is_connected()):
            cursor.close()
            connection.close()
    if(time.minute==0 and time.second==0):
        try:
            connection = mysql.connector.connect(host='localhost', database='Weather', user='root', password='root')
            if connection.is_connected():
                cursor = connection.cursor()
                cursor.execute("select database();")
                record = cursor.fetchone()
                print("You're connected to database: ", record)
                sql = "insert into hourly (Datetime, temp, hump, weather) values ('%s',%s,%s,%s);" % (timeHourly,temp,hump,weather)
                cursor.execute(sql)
                connection.commit()
        except Error as e:
            print("Error while connecting to MySQL", e)
        finally:
            if (connection.is_connected()):
                cursor.close()
                connection.close()
                print("MySQL connection is closed")
    if(time.hour==23 and time.minute == 59 and time.second==59):       
        try:
            connection = mysql.connector.connect(host='localhost', database='Weather', user='root', password='root')
            if connection.is_connected():
                cursor = connection.cursor()
                cursor.execute("select database();")
                record = cursor.fetchone()
                print("You're connected to database: ", record)
                sql = "insert into daily (date, max_temp, min_temp) values ('%s',%s,%s);" % (timeDaily,max_temp,min_temp)
                cursor.execute(sql)
                connection.commit()      
        except Error as e:
            print("Error while connecting to MySQL", e)
        finally:
            if (connection.is_connected()):
                cursor.close()
                connection.close()
                print("MySQL connection is closed")
        max_temp = 0
        min_temp = 100