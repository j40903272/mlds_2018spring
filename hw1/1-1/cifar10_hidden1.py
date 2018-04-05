
# coding: utf-8

# In[1]:


import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam


# In[2]:


(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# In[3]:


model = Sequential()
model.add(Conv2D(5, (5, 5), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(22))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()


# In[4]:


model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])


# In[5]:


history = model.fit(x_train, y_train,
              batch_size=512,
              epochs=50,
              validation_data=(x_test, y_test))


# In[6]:


import numpy as np
np.save('cifar_1_acc', history.history['acc'])
np.save('cifar_1_loss', history.history['loss'])

