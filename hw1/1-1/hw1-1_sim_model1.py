import numpy as np
from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD
import math
import matplotlib.pyplot as plt
import pickle

def build_model1():
	model = Sequential()
	model.add(Dense(5, activation='relu', input_shape=(1,)))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(10, activation='relu'))
<<<<<<< HEAD
	model.add(Dropout(0.25))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(5, activation='relu'))
	model.add(Dense(1, activation='relu'))
=======
	model.add(Dense(10, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(5, activation='relu'))
	model.add(Dense(1, activation='linear'))
>>>>>>> 77
	return model

def main():
	adam = Adam(lr=0.001, decay=1e-5, clipvalue=1e-5)
	model1 = build_model1()
	model1.compile(optimizer='adam', loss='mse')
	
<<<<<<< HEAD
	train_x = np.load('train.npy')
	print(train_x)
	train_y = []
	for i in range(train_x.shape[0]):
		train_y.append(float(train_x[i]*train_x[i]*train_x[i]*train_x[i] + train_x[i]*train_x[i] + train_x[i] + 1))

	train_y = np.array(train_y)
	history1 = model1.fit(train_x, train_y, epochs=100, batch_size=1000, verbose=1)
=======
	train_x = np.random.rand(10000)
	np.save('train.npy', train_x)
	print(train_x)
	train_y = []
	for i in range(train_x.shape[0]):
		train_y.append(float(math.sin(5*math.pi*train_x[i])/(5*math.pi*train_x[i])))

	train_y = np.array(train_y)
	history1 = model1.fit(train_x, train_y, epochs=10000, batch_size=1000, verbose=1)
>>>>>>> 77
	model1.save('deep.h5')
	loss1 = history1.history['loss']
	pickle.dump(loss1, open('deep.pkl', 'wb'))
	print(train_x)
	#loss1 = pickle.load(open('deep.pkl', 'rb')

if __name__ == "__main__":
    main()
