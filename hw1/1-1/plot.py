import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
import os

def main():
	loss1 = pickle.load(open('model1.pkl','rb'))
	loss2 = pickle.load(open('model2.pkl','rb'))
	loss3 = pickle.load(open('model3.pkl','rb'))
	x = np.arange(10000)
	plt.plot(x, loss1)
	plt.plot(x, loss2)
	plt.plot(x, loss3)
	plt.title('Simulate function')
	plt.ylabel('loss')
	plt.xlabel('epochs')
	plt.yscale('log')
	plt.legend(['Model1','Model2', 'Model3'], loc='upper right')
	plt.show()

if __name__ == '__main__':
	main()