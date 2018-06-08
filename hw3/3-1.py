import os
import glob
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
from sklearn.utils import shuffle
import time
import cv2
import scipy
import imageio
from PIL import Image
import matplotlib.gridspec as gridspec
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.layers import Input
from keras.layers import Conv2D, Conv2DTranspose, Dropout
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.advanced_activations import LeakyReLU
from GAN import get_disc_normal, get_gen_normal
import matplotlib.pyplot as plt
from keras.models import load_model
import keras.backend as K
from scipy.interpolate import spline
K.set_image_dim_ordering('tf')

from collections import deque

np.random.seed(1337)

def norm_img(img):
    img = (img / 127.5) - 1
    return img

def denorm_img(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8) 

def sample_from_dataset(batch_size, image_shape, data_dir=None, data = None):
    sample_dim = (batch_size,) + image_shape
    sample = np.empty(sample_dim, dtype=np.float32)
    all_data_dirlist = list(glob.glob(data_dir))
    sample_imgs_paths = np.random.choice(all_data_dirlist,batch_size)
    for index,img_filename in enumerate(sample_imgs_paths):
        image = Image.open(img_filename)
        #print(image.size)
        image = image.resize(image_shape[:-1])
        image = image.convert('RGB') #remove transparent ('A') layer
        #print(image.size)
        #print('\n')
        image = np.asarray(image)
        image = norm_img(image)
        sample[index,...] = image
    return sample

def gen_noise(batch_size, noise_shape):
    #input noise to gen seems to be very important!
    return np.random.normal(0, 1, size=(batch_size,)+noise_shape)

def generate_images(generator, save_dir):
    noise = gen_noise(batch_size,noise_shape)
    #using noise produced by np.random.uniform - the generator seems to produce same image for ANY noise - 
    #but those images (even though they are the same) are very close to the actual image - experiment with it later.
    fake_data_X = generator.predict(noise)
    print("Displaying generated images")
    plt.figure(figsize=(5,5))
    gs1 = gridspec.GridSpec(5, 5)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(fake_data_X.shape[0],25,replace=False)
    for i in range(25):
        #plt.subplot(4, 4, i+1)
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = fake_data_X[rand_index, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(save_dir+str(time.time())+"_GENERATEDimage.png",bbox_inches='tight',pad_inches=0)
    #plt.show()

def save_imgs(generator, noise_shape):
    import matplotlib.pyplot as plt
    r, c = 5, 5
    noise = gen_noise(r*c,noise_shape)
    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator.predict(noise)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("output.png")
    plt.close()

def clip_weight(discriminator):
    weights = [np.clip(w, -0.01, 0.01) for w in discriminator.get_weights()]
    discriminator.set_weights(weights)

def save_img_batch(img_batch,img_save_dir):
    plt.figure(figsize=(5,5))
    gs1 = gridspec.GridSpec(5, 5)
    gs1.update(wspace=0, hspace=0)
    rand_indices = np.random.choice(img_batch.shape[0],25,replace=False)
    #print(rand_indices)
    for i in range(25):
        #plt.subplot(4, 4, i+1)
        ax1 = plt.subplot(gs1[i])
        ax1.set_aspect('equal')
        rand_index = rand_indices[i]
        image = img_batch[rand_index, :,:,:]
        fig = plt.imshow(denorm_img(image))
        plt.axis('off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(img_save_dir,bbox_inches='tight',pad_inches=0)
    #plt.show()   



noise_shape = (1,1,100)
num_steps = 25000
batch_size = 64

img_save_dir = "generate_image/"
model_g = "generator.h5"
model_d = "discriminator.h5"

save_model = True



#image_shape = (96,96,3)
image_shape = (64,64,3)
data_dir =  "faces/*.jpg"
log_dir = img_save_dir
save_model_dir = "model/"

discriminator = get_disc_normal(image_shape)
generator = get_gen_normal(noise_shape)

discriminator.trainable = False

opt = Adam(lr=0.00015, beta_1=0.5) #same as gen
gen_inp = Input(shape=noise_shape)
GAN_inp = generator(gen_inp)
GAN_opt = discriminator(GAN_inp)
gan = Model(input = gen_inp, output = GAN_opt)
gan.compile(loss = 'binary_crossentropy', optimizer = opt, metrics=['accuracy'])
gan.summary()

avg_disc_fake_loss = deque([0], maxlen=250)     
avg_disc_real_loss = deque([0], maxlen=250)
avg_GAN_loss = deque([0], maxlen=250)
#x_train = np.load(open('image.npy', 'rb'))

for step in range(num_steps): 
    itr_d = 3
    tot_step = step
    #print("Begin step: ", tot_step)
    step_begin_time = time.time() 
    #idx = np.random.choice(x_train.shape[0], batch_size, replace=False)
    #real_data_X = x_train[idx]
    real_data_X = sample_from_dataset(batch_size, image_shape, data_dir = data_dir)
    noise = gen_noise(batch_size,noise_shape)    
    fake_data_X = generator.predict(noise)
    
    if (tot_step % 100) == 0:
        step_num = str(tot_step).zfill(4)
        save_img_batch(fake_data_X,img_save_dir+step_num+"_image.png")

        
    #data_X = np.concatenate([real_data_X,fake_data_X])
    
    real_data_Y = np.ones(batch_size) - (np.random.random_sample(batch_size)*0.4 - 0.2)
    fake_data_Y = np.random.random_sample(batch_size)*0.2
     
    #data_Y = np.concatenate((real_data_Y,fake_data_Y))
        
    discriminator.trainable = True
    generator.trainable = False
    #train discriminator itr_d times
    #for i in range(itr_d):
    #   	real_data_X = sample_from_dataset(batch_size, image_shape, data_dir = data_dir)
    #    noise = gen_noise(batch_size,noise_shape)    
    #    fake_data_X = generator.predict(noise)
    dis_metrics_real = discriminator.train_on_batch(real_data_X,real_data_Y)   #training seperately on real
    dis_metrics_fake = discriminator.train_on_batch(fake_data_X,fake_data_Y)   #training seperately on fake
    avg_disc_fake_loss.append(dis_metrics_fake[0])
    avg_disc_real_loss.append(dis_metrics_real[0])
    
    generator.trainable = True

    GAN_X = gen_noise(batch_size,noise_shape)

    GAN_Y = real_data_Y
    
    discriminator.trainable = False
    
    gan_metrics = gan.train_on_batch(GAN_X,GAN_Y)
    print("step: %d Disc: real loss: %f fake loss: %f GAN loss: %f" % (tot_step, dis_metrics_real[0], dis_metrics_fake[0], gan_metrics[0]))
    
    avg_GAN_loss.append(gan_metrics[0])
    
        
    end_time = time.time()
    diff_time = int(end_time - step_begin_time)
    #print("Step %d completed. Time took: %s secs." % (tot_step, diff_time))

    if ((tot_step+1) % 500) == 0:
        print("-----------------------------------------------------------------")
        print("Average Disc_fake loss: %f" % (np.mean(avg_disc_fake_loss)))    
        print("Average Disc_real loss: %f" % (np.mean(avg_disc_real_loss)))    
        print("Average GAN loss: %f" % (np.mean(avg_GAN_loss)))
        print("-----------------------------------------------------------------")
        #discriminator.trainable = True
        #generator.trainable = True
        #generator.save(save_model_dir+str(tot_step)+"_GENERATOR.h5")
        #discriminator.save(save_model_dir+str(tot_step)+"_DISCRIMINATOR.h5")
text_file = open(log_dir+"\\training_log.txt", "w")
text_file.write("Average Disc_fake loss: %f\n" % (np.mean(avg_disc_fake_loss)))
text_file.write("Average Disc_real loss: %f\n" % (np.mean(avg_disc_real_loss)))
text_file.write("Average GAN loss: %f\n" % (np.mean(avg_GAN_loss)))
text_file.close()

discriminator.trainable = True
generator.trainable = True
generator.save(save_model_dir+model_g)
discriminator.save(save_model_dir+model_d)
'''
#Generating GIF from PNG
images = []
all_data_dirlist = list(glob.glob(img_save_dir+"*_image.png"))
for filename in all_data_dirlist:
    img_num = filename.split('\\')[-1][0:-10]
    if (int(img_num) % 100) == 0:
        images.append(imageio.imread(filename))
imageio.mimsave(img_save_dir+'movie.gif', images)
'''