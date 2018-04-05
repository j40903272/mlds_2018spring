import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Dropout, Input, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import adam
import matplotlib.pyplot as plt
import pickle

def build_model(num):
	n = int(num*10)
	model = Sequential()
	model.add(Conv2D(8,3,3, activation='relu', input_shape=(32,32,3)))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(16,3,3, activation='relu'))
	model.add(AveragePooling2D(pool_size=(2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())	
	model.add(Dense(n, activation='relu'))
	model.add(Dense(n, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	return model

def main():
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()
	train_x = x_train/255
	test_x = x_test/255
	train_y = np_utils.to_categorical(y_train)
	test_y = np_utils.to_categorical(y_test)
	train_x = train_x.reshape(train_x.shape[0],32,32,3)
	test_x = test_x.reshape(test_x.shape[0],32,32,3)
	parameter = np.arange(200, 9, -10)
	params = []
	loss = []
	val_loss = []
	acc = []
	val_acc = []
	for p in parameter:
		model = build_model(p)
		num_param = model.count_params()
		params.append(num_param)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		history = model.fit(train_x, train_y, epochs=50, batch_size=1000, shuffle=True, validation_data=(test_x, test_y))
		## plot every model as a point
		#plt.plot([num_param], [history.history['loss'][-1]], marker='o', markersize=3, color="red")
		#plt.plot([num_param], [history.history['val_loss'][-1]], marker='o', markersize=3, color="blue")
		## store this model's loss and accuracy
		#loss.append(min(history.history['loss']))
		#val_loss.append(min(history.history['val_loss']))
		#acc.append(max(history.history['acc']))
		#val_acc.append(max(history.history['val_acc']))
	## store num of params , loss, and accuracy
	#pickle.dump(params, open('params.pkl', 'wb'))
	#pickle.dump(loss, open('loss.pkl', 'wb'))
	#pickle.dump(val_loss, open('val_loss.pkl', 'wb'))
	#pickle.dump(acc, open('acc.pkl', 'wb'))
	#pickle.dump(val_acc, open('val_acc.pkl', 'wb'))
	## plot the image
	#plt.title('Number of Parameters Test')
	#plt.ylabel('loss')
	#plt.xlabel('parameters')
	#plt.legend(['Train','Test'], loc='upper right')
	#plt.show()


if __name__ == '__main__':
	main()