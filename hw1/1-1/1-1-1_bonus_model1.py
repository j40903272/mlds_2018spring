
# coding: utf-8

# In[1]:


import numpy as np
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam, SGD
import math
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


x = [ [i] for i in np.linspace(0, 5, 1000)]
y = [ i*np.cos(i) for i in x ]
x = np.array(x)
y = np.array(y)


# In[3]:


model = Sequential()
model.add(Dense(5, activation='relu', input_shape=(1,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='linear'))
model.add(Dense(1, activation='linear'))
model.summary()


# In[4]:


model.compile(optimizer=Adam(), loss='mse')


# In[5]:


history = model.fit(x, y, epochs=1000, batch_size=100)


# In[6]:


loss = history.history['loss']
np.save('model1_loss', loss)
y_pred = model.predict(x)
np.save('model1_pred', y_pred)


# In[7]:


plt.plot(loss)

