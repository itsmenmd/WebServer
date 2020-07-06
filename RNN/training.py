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
from nomalize import dataset, datasetnan, max_temp, min_temp


###########################-DEVIDE TRAIN DATA-#############################3
TRAIN_SPLIT = 25000
TRAIN_SPLIT2 = 1000
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

x_train_temp, y_train_temp = variate_data(dataset,dataset[:,0], 0, TRAIN_SPLIT, past_history, future_target, multi=True)
x_val_temp , y_val_temp = variate_data(dataset[:,[0,1]],dataset[:,0], TRAIN_SPLIT, None, past_history, future_target, multi=True)

x_train_hump, y_train_hump = variate_data(dataset,dataset[:,1], 0, TRAIN_SPLIT, past_history, future_target, multi=True)
x_val_hump , y_val_hump = variate_data(dataset[:,[0,1]],dataset[:,1], TRAIN_SPLIT, None, past_history, future_target, multi=True)

x_train_ww = dataset[:TRAIN_SPLIT,[0,1]]
y_train_ww = datasetnan[:TRAIN_SPLIT]
x_val_ww = dataset[TRAIN_SPLIT:len(dataset),[0,1]]
y_val_ww = datasetnan[TRAIN_SPLIT:len(dataset)]

past_history2 = 70
future_target2 = 7

x_train_temp_max, y_train_temp_max = variate_data(max_temp, max_temp, 0, TRAIN_SPLIT2, past_history2, future_target2)
x_val_temp_max , y_val_temp_max = variate_data(max_temp, max_temp, TRAIN_SPLIT2, None, past_history2, future_target2)

x_train_temp_min, y_train_temp_min = variate_data(min_temp, min_temp, 0, TRAIN_SPLIT2, past_history2, future_target2)
x_val_temp_min , y_val_temp_min = variate_data(min_temp, min_temp, TRAIN_SPLIT2, None, past_history2, future_target2)

######################################-SHOW PLOT-###################################

def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps

def show_plot(true_future, prediction):
  plt.figure(figsize=(12, 6))
#   num_in = create_time_steps(len(history))
  num_out = len(true_future)

#   plt.plot(num_in, np.array(history), label='History')
  plt.plot(np.arange(num_out), np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out), np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()

###################################-CHANGE TO 3D DATA-################################

BATCH_SIZE = 256
BUFFER_SIZE = 1000

train_data_temp = tf.data.Dataset.from_tensor_slices((x_train_temp, y_train_temp))
train_data_temp = train_data_temp.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_data_temp = tf.data.Dataset.from_tensor_slices((x_val_temp, y_val_temp))
val_data_temp = val_data_temp.batch(BATCH_SIZE).repeat()

train_data_hump = tf.data.Dataset.from_tensor_slices((x_train_hump, y_train_hump))
train_data_hump = train_data_hump.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
val_data_hump = tf.data.Dataset.from_tensor_slices((x_val_hump, y_val_hump))
val_data_hump = val_data_hump.batch(BATCH_SIZE).repeat()

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

###############################-COMPLIE MODEL-##################################

temp_model = tf.keras.models.Sequential()
temp_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_temp[1,:,:].shape))
temp_model.add(tf.keras.layers.Dense(12))
temp_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

hump_model = tf.keras.models.Sequential()
hump_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_hump[1,:,:].shape))
hump_model.add(tf.keras.layers.Dense(12))
hump_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

ww_model = tf.keras.models.Sequential([ 
  tf.keras.layers.Dense(64, activation='relu', input_dim=2),
  tf.keras.layers.Dense(3, activation='softmax')
])
ww_model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

max_model = tf.keras.models.Sequential()
max_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_temp_max.shape[-2:]))
max_model.add(tf.keras.layers.Dense(7))
max_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

min_model = tf.keras.models.Sequential()
min_model.add(tf.keras.layers.LSTM(32, input_shape=x_train_temp_min.shape[-2:]))
min_model.add(tf.keras.layers.Dense(7))
min_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

#########################-TRAIN DATA-##################################

EVALUATION_INTERVAL = 200
EPOCHS = 10

# temp_history = temp_model.fit(train_data_temp, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data_temp, validation_steps=50)
# temp_model.save('temp_model.h5')

# hump_history = hump_model.fit(train_data_hump, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data_hump, validation_steps=50)
# hump_model.save('hump_model.h5')

# max_history = max_model.fit(train_data_temp_max, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data_temp_max, validation_steps=50)
# max_model.save('max_model.h5')

# min_history = min_model.fit(train_data_temp_min, epochs=EPOCHS, steps_per_epoch=EVALUATION_INTERVAL, validation_data=val_data_temp_min, validation_steps=50)
# min_model.save('min_model.h5')


# ww_model.fit(x_train_ww, y_train_ww, epochs=10)
# ww_model.save('weather_model.h5')
# ww_model = tf.keras.models.load_model('weather_model.h5')
# test_loss, test_acc = ww_model.evaluate(x_val_ww,  y_val_ww, verbose=2)

# print('\nTest accuracy:', test_acc)
# predictions = ww_model.predict(x_val_ww)
# print("Gia tri thuc: ",y_val_ww[152])
# print(predictions[152])
# print("Gia tri du doan: ",np.argmax(predictions[152]))

# temp_model = tf.keras.models.load_model('min_model.h5')
# for x, y in val_data_temp.take(1):
#     print(np.argmax(temp_model.predict(x)[0]))
    
# def plot_train_history(history, title):
#   loss = history.history['loss']
#   val_loss = history.history['val_loss']

#   epochs = range(len(loss))

#   plt.figure()

#   plt.plot(epochs, loss, 'b', label='Training loss')
#   plt.plot(epochs, val_loss, 'r', label='Validation loss')
#   plt.title(title)
#   plt.legend()

#   plt.show()

# plot_train_history(max_history,'Single Step Training and validation loss')
# plot_train_history(min_history,'Single Step Training and validation loss')
# for x, y in val_data_temp.take(3):
#   show_plot(x[0], y[0], temp_model.predict(x)[0])

#############################-SHOW PREDICTION DEMO-###################################################
temp_model = tf.keras.models.load_model('temp_model.h5')
for x, y in val_data_temp.take(3):
  show_plot(y[0],temp_model.predict(x)[0])
temp_model = tf.keras.models.load_model('max_model.h5')
for x, y in val_data_temp_max.take(3):
  show_plot(y[0],temp_model.predict(x)[0])
temp_model = tf.keras.models.load_model('min_model.h5')
for x, y in val_data_temp_min.take(3):
  show_plot(y[0],temp_model.predict(x)[0])
  

# predict = temp_model.predict(val_data_temp.take(1))
# predict = predict*temp_std + temp_mean
# print(predict)
