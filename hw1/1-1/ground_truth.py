import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
import os
import math
def main():
	model1 = load_model(os.path.join(os.path.dirname(__file__),'model1.h5'))
	model2 = load_model(os.path.join(os.path.dirname(__file__),'model2.h5'))
	model3 = load_model(os.path.join(os.path.dirname(__file__),'model3.h5'))
	train_x = np.load('train.npy')
	train_y = []
	x = np.arange(0, 1, 0.00001)
	for i in range(x.shape[0]):
		train_y.append(float(math.sin(5*math.pi*x[i])/(5*math.pi*x[i])))
	ans1 = model1.predict(x)
	ans2 = model2.predict(x)
	ans3 = model3.predict(x)
	plt.plot(x, train_y)
	plt.plot(x, ans1)
	plt.plot(x, ans2)
	plt.plot(x, ans3)
	plt.title('Simulate function')
	plt.ylabel('loss')
	plt.xlabel('epochs')
	plt.legend(['Ground Truth', 'Deep','Shallow', 'Most shallow'], loc='upper right')
	plt.show()

if __name__ == '__main__':
	main()