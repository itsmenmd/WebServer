import time
import paho.mqtt.client as mqtt
import random,json
from save_2_db import storeDB
from datetime import datetime

MQTT_Broker = "localhost"
MQTT_Port = 1883
Keep_Alive_Interval = 45
MQTT_Topic = "home/sensor"

def on_connect(client, userdata, flags, rc):
        if rc != 0:
                pass
                print("Unable to connect to MQTT Broker,,,")
        else:
                print("Connected with MQTT Broker: " + str(MQTT_Broker))
        client.subscribe(MQTT_Topic,0)

def on_message(client, userdata, msg):
        print("MQTT Data Received...")
        print("MQTT Toic: " + msg.topic)
        print("Data: " + str(msg.payload.decode("utf-8")))
        print("")
        storeDB(msg.payload)
client = mqtt.Client()
client.username_pw_set(username="client3")
client.on_connect = on_connect
# client.subscribe(MQTT_Topic,0) 
client.on_message = on_message          #attach function to callback
client.connect(MQTT_Broker, MQTT_Port, Keep_Alive_Interval) 
   #connect to brocket
client.loop_forever()
