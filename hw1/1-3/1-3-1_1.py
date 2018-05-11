import numpy as np
from keras.utils import np_utils
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import adam
import matplotlib.pyplot as plt
import pickle

def build_model():
	model = Sequential()
	model.add(Conv2D(32,3,3, activation='relu', input_shape=(28,28,1)))
	model.add(Conv2D(64,3,3, activation='relu'))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(10, activation='softmax'))
	return model

def main():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	np.random.shuffle(y_train)
	pickle.dump(y_train, open('test.pkl', 'wb'))
	train_x = x_train/255
	test_x = x_test/255
	train_y = np_utils.to_categorical(y_train)
	test_y = np_utils.to_categorical(y_test)
	train_x = train_x.reshape(train_x.shape[0],28,28,1)
	test_x = test_x.reshape(test_x.shape[0],28,28,1)
	print(train_y[0])
	model = build_model()
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()
	history = model.fit(train_x, train_y, epochs=2000, batch_size=1024, validation_data=(test_x, test_y))
	#pickle.dump(history, open('history.pkl', 'wb'))
	model.save('hw1-3.h5')
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Random label test')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['Train','Test'], loc='upper left')
	plt.show()


if __name__ == '__main__':
	main()