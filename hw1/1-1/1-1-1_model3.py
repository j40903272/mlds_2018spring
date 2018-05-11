import numpy as np
from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD
import math
import matplotlib.pyplot as plt
import pickle

def build_model3():
	model = Sequential()
	model.add(Dense(20, activation='relu', input_dim=1))
	model.add(Dense(19, activation='relu'))
	model.add(Dense(1, activation='linear'))
	return model

def main():
	adam = Adam(lr=0.001, decay=1e-5, clipvalue=1e-5)

	model3 = build_model3()
	model3.compile(optimizer='adam', loss='mse')
	
	train_x = np.load('train.npy')
	train_y = []
	for i in range(train_x.shape[0]):
		train_y.append(float(math.sin(5*math.pi*train_x[i])/(5*math.pi*train_x[i])))

	train_y = np.array(train_y)
	history3 = model3.fit(train_x, train_y, epochs=10000, batch_size=1000, verbose=1)
	model3.save('most_shallow.h5')
	loss3 = history3.history['loss']
	pickle.dump(loss3, open('most shallow.pkl', 'wb'))
	print(train_x)

if __name__ == "__main__":
    main()
