#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from keras.layers import Dense, LSTM, Bidirectional, Dropout, Lambda
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('weatherHistory.csv', parse_dates=['Formatted Date'])
df.head()


# In[2]:


df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)


# In[3]:


df.info()


# **Data** **Cleaning**

# In[4]:


df.isna().sum()


# In[5]:


df.shape


# In[6]:


batas_mae = (df['Wind Speed (km/h)'].max() - df['Wind Speed (km/h)'].min()) * 0.1
batas_mae


# **Mengambil 20.000 data dan membaginya untuk training dan validation**

# In[7]:


df_train = df[1:16001]
df_val = df[16002:20002]


# In[8]:


plt.figure(figsize=(15,5))
plt.plot(df_train.index, df_train[['Wind Speed (km/h)']])
plt.title('Wind Speed Average',
          fontsize=20);
plt.plot(df_val.index, df_val[['Wind Speed (km/h)']])
plt.title('Wind Speed Average',
          fontsize=20);


# In[9]:


minmaxscaler = MinMaxScaler(feature_range = (0,1))
scale_train = minmaxscaler.fit_transform(df_train[['Wind Speed (km/h)']])
df_train[['Wind Speed (km/h)']] = scale_train

scale_val = minmaxscaler.fit_transform(df_val[['Wind Speed (km/h)']])
df_val[['Wind Speed (km/h)']] = scale_val


# In[10]:


date_train = df_train['Formatted Date']
wind_train  = df_train['Wind Speed (km/h)'].values

date_val = df_val['Formatted Date']
wind_val = df_val['Wind Speed (km/h)'].values

date_train.shape


# In[11]:


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    return ds.batch(batch_size).prefetch(1)


# **Modelling**

# In[12]:


train_set = windowed_dataset(wind_train, window_size=60, batch_size=100, shuffle_buffer=1000)
test_set  = windowed_dataset(wind_val, window_size=60, batch_size=100, shuffle_buffer=1000)
model = tf.keras.models.Sequential([
                                    tf.keras.layers.LSTM(64, return_sequences=True),
                                    tf.keras.layers.LSTM(64),
                                    tf.keras.layers.Dense(30, activation='relu'),
                                    tf.keras.layers.Dropout(0.2),
                                    tf.keras.layers.Dense(30, activation='relu'),
                                    tf.keras.layers.Dropout(0.5),
                                    tf.keras.layers.Dense(1)
])


# **Membuat Class Callback**

# In[13]:


from tensorflow.keras.callbacks import Callback
class myCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('mae')<0.09 and logs.get('val_mae')<0.09):
            print("MAE has reached below 10%")
            self.model.stop_training = True
callbacks = myCallback()


# **Melatih Model**

# In[14]:


optimizer = tf.keras.optimizers.SGD(learning_rate=1.0000e-04, momentum=0.9)
model.compile(loss=tf.keras.losses.Huber(),
              optimizer=optimizer,
              metrics=["mae"])
history = model.fit(train_set,
                    epochs=700,
                    callbacks=[callbacks],
                    validation_data=test_set)


# In[15]:


model.evaluate(test_set)


# In[16]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()


# In[17]:


plt.plot(history.history['mae'])
plt.title('Model accuracy')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='lower right')
plt.show()


# In[ ]:




