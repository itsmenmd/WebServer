import time 
import paho.mqtt.client as mqtt
import random, json
from datetime import datetime

MQTT_Broker = "localhost"

MQTT_Port = 1883

Keep_Alive_Interval = 45

MQTT_Topic1 = "home/sensor"



def on_connect(client, userdata, rc):

    if rc != 0:

        pass

        print("Unable to connect to MQTT Broker,,,")

    else:

        print("Connected with MQTT Broker: " + str(MQTT_Broker))



def on_publish(client, userdata, mid):

    pass



def on_disconnect(client, userdata, rc):

    if rc != 0:

        pass



mqttc = mqtt.Client("client1")

# mqttc.username_pw_set(username="client1")

# mqttc.on_connect = on_connect

# mqttc.on_disconnect = on_disconnect

# mqttc.on_publish = on_publish

mqttc.connect(MQTT_Broker, MQTT_Port, Keep_Alive_Interval)



def pusblish_2_topic(topic, message):

    mqttc.publish(topic,message)

    print(("Published: " + str(message) + " " + "on MQTT Topic: " + str(topic)))

    print("")



def publish_fake_sensor1_values_2_MQTT():

    Temp_fake_value = int(random.randint(24, 25))

    Hum_fake_value = int(random.randint(78, 80))

    testlist = [0]
    ww_fake_value = random.choice(testlist)

    Sensor_data = {}

    Sensor_data['temp'] = Temp_fake_value

    Sensor_data['hum'] = Hum_fake_value

    Sensor_data['ww'] = ww_fake_value

    
    sensor_json_data = json.dumps(Sensor_data)

    # print("Publishing Sensor data: ")

    pusblish_2_topic(MQTT_Topic1, sensor_json_data)




while True:

    publish_fake_sensor1_values_2_MQTT()


    time.sleep(3)