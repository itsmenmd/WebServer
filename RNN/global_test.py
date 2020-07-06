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
    print("max_temp: ", max_temp, ", min_temp: ",min_temp)
    

def testCall(section):
    print(section)
