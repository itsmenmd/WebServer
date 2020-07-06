from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np 
import os
import statistics
import mysql.connector 
from mysql.connector import Error

try:
    connection = mysql.connector.connect(host='localhost', database='Weather', user='root', password='22081997')
    if connection.is_connected():
        cursor = connection.cursor()
        cursor.execute("select database();")
        record = cursor.fetchone()
        print("You're connected to database: ", record)
        sql = "select max_temp,min_temp from test3;"
        cursor.execute(sql)
        result = cursor.fetchall()
        max_temp = []
        min_temp = []
        for row in result:
            max_temp.append(row[0])
            min_temp.append(row[1])
        max_temp = np.asarray(max_temp)
        min_temp = np.asarray(min_temp)
except Error as e:
    print("Error while connecting to MySQL", e)
finally:
    if (connection.is_connected()):
        cursor.close()
        connection.close()
        print("MySQL connection is closed")

TRAIN_SPLIT2 = 1000
max_temp_mean = max_temp[:TRAIN_SPLIT2].mean()
max_temp_std = max_temp[:TRAIN_SPLIT2].std()
min_temp_mean = min_temp[:TRAIN_SPLIT2].mean(axis=0)
min_temp_std = min_temp[:TRAIN_SPLIT2].std(axis=0)
max_temp = (max_temp - max_temp_mean)/max_temp_std
min_temp = (min_temp - min_temp_mean)/min_temp_std

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

past_history2 = 70
future_target2 = 7

x_train_temp_max, y_train_temp_max = variate_data(max_temp, max_temp, 0, TRAIN_SPLIT2, past_history2, future_target2)
x_val_temp_max , y_val_temp_max = variate_data(max_temp, max_temp, TRAIN_SPLIT2, None, past_history2, future_target2)

x_train_temp_min, y_train_temp_min = variate_data(min_temp, min_temp, 0, TRAIN_SPLIT2, past_history2, future_target2)
x_val_temp_min , y_val_temp_min = variate_data(min_temp, min_temp, TRAIN_SPLIT2, None, past_history2, future_target2)

def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps

def show_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history), label='History')
  plt.plot(np.arange(num_out), np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out), np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()

BATCH_SIZE2 = 64
BUFFER_SIZE2 = 200
train_data_temp_max = tf.data.Dataset.from_tensor_slices((x_train_temp_max, y_train_temp_max))
train_data_temp_max = train_data_temp_max.cache().shuffle(BUFFER_SIZE2).batch(BATCH_SIZE2).repeat()
val_data_temp_max = tf.data.Dataset.from_tensor_slices((x_val_temp_max, y_val_temp_max))
val_data_temp_max = val_data_temp_max.batch(BATCH_SIZE2).repeat()

train_data_temp_min = tf.data.Dataset.from_tensor_slices((x_train_temp_min, y_train_temp_min))
train_data_temp_min = train_data_temp_min.cache().shuffle(BUFFER_SIZE2).batch(BATCH_SIZE2).repeat()
val_data_temp_min = tf.data.Dataset.from_tensor_slices((x_val_temp_min, y_val_temp_min))
val_data_temp_min = val_data_temp_min.batch(BATCH_SIZE2).repeat()

# max_model = tf.keras.models.Sequential()
# max_model.add(tf.keras.layers.LSTM(8, input_shape=x_train_temp_max.shape[-2:]))
# max_model.add(tf.keras.layers.Dense(7))
# max_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

# min_model = tf.keras.models.Sequential()
# min_model.add(tf.keras.layers.LSTM(8, input_shape=x_train_temp_min.shape[-2:]))
# min_model.add(tf.keras.layers.Dense(7))
# min_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

# EVALUATION_INTERVAL = 200
# EPOCHS = 10
# max_history = max_model.fit(train_data_temp_max, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data_temp_max, validation_steps=50)
# max_model.save('max_model.h5')

# min_history = min_model.fit(train_data_temp_min, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data_temp_min, validation_steps=50)
# min_model.save('min_model.h5')

# for x, y in val_data_temp_max.take(3):
#   show_plot(x[0], y[0], max_model.predict(x)[0])

# for x, y in val_data_temp_min.take(3):
#   show_plot(x[0], y[0], min_model.predict(x)[0])