import numpy as np
import os
import sys
from skimage import io
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import matplotlib.gridspec as gridspec
import torchvision
from keras.models import load_model

np.random.seed(7787)

def denorm_img(img):
	img = (img + 1) * 127.5
	return img.astype(np.uint8)

def save_imgs(generator):
	import matplotlib.pyplot as plt
	r, c = 5, 5
	noise = gen_noise(r*c, (1,1,100))
	# gen_imgs should be shape (25, 64, 64, 3)
	gen_imgs = generator.predict(noise)
	fig, axs = plt.subplots(r, c)
	cnt = 0
	for i in range(r):
		for j in range(c):
			axs[i,j].imshow(denorm_img(gen_imgs[cnt, :,:,:]))
			axs[i,j].axis('off')
			cnt += 1
	fig.savefig("samples/gan.png")
	plt.close()

def gen_noise(batch_size, noise_shape):
	#input noise to gen seems to be very important!
	return np.random.normal(0, 1, size=(batch_size,)+noise_shape)

model = load_model('model/generator.h5')
save_imgs(model)