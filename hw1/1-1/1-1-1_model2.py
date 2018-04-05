import numpy as np
from keras.layers import Dense, Input, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, SGD
import math
import matplotlib.pyplot as plt
import pickle


def build_model2():
	model = Sequential()
	model.add(Dense(7, activation='relu', input_shape=(1,)))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(19, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1, activation='linear'))
	return model

def main():
	adam = Adam(lr=0.001, decay=1e-5, clipvalue=1e-5)
	model2 = build_model2()
	model2.compile(optimizer='adam', loss='mse')
	
	train_x = np.load('train.npy')
	train_y = []
	for i in range(train_x.shape[0]):
		train_y.append(float(math.sin(5*math.pi*train_x[i])/(5*math.pi*train_x[i])))

	train_y = np.array(train_y)
	history2 = model2.fit(train_x, train_y, epochs=10000, batch_size=1000, verbose=1)
	model2.save('shallow.h5')
	loss2 = history2.history['loss']
	pickle.dump(loss2, open('shallow.pkl', 'wb'))
	print(train_x)
	#loss2 = pickle.load(open('shallow.pkl', 'rb')

if __name__ == "__main__":
    main()
