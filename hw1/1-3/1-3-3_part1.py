
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


w_conv_1 = np.load('model_64/w_conv.npy')
b_conv_1 = np.load('model_64/b_conv.npy')
w_fc1_1 = np.load('model_64/w_fc1.npy')
b_fc1_1 = np.load('model_64/b_fc1.npy')
w_fc2_1 = np.load('model_64/w_fc2.npy')
b_fc2_1 = np.load('model_64/b_fc2.npy')

w_conv_2 = np.load('model_1024/w_conv.npy')
b_conv_2 = np.load('model_1024/b_conv.npy')
w_fc1_2 = np.load('model_1024/w_fc1.npy')
b_fc1_2 = np.load('model_1024/b_fc1.npy')
w_fc2_2 = np.load('model_1024/w_fc2.npy')
b_fc2_2 = np.load('model_1024/b_fc2.npy')


# In[3]:


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 


# In[4]:


x = tf.placeholder(tf.float32, [None, 784], name='x')
y = tf.placeholder(tf.float32, [None, 10], name='y')
alpha = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = tf.placeholder(tf.float32, [5, 5, 1, 32])
b_conv1 = tf.placeholder(tf.float32, [1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_fc1 = tf.placeholder(tf.float32, [14*14*32, 64])
b_fc1 = tf.placeholder(tf.float32, [64])
h_pool1_flat = tf.reshape(h_pool1, [-1, 14*14*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = tf.placeholder(tf.float32, [64, 10])
b_fc2 = tf.placeholder(tf.float32, [10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y_conv')

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv+1e-5), reduction_indices=[1]), name='cross_entropy')

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[5]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[6]:


def get_acc_loss(alpha):
    with tf.Session() as sess:
        
        train_dict = [{
            x: mnist.train.images[1000*i:1000*(i+1)],
            y: mnist.train.labels[1000*i:1000*(i+1)],
            keep_prob: 1.0,
            W_conv1: (1-alpha)*w_conv_1+alpha*w_conv_2,
            b_conv1: (1-alpha)*b_conv_1+alpha*b_conv_2,
            W_fc1:   (1-alpha)*w_fc1_1+alpha*w_fc1_2,
            b_fc1:   (1-alpha)*b_fc1_1+alpha*b_fc1_2,
            W_fc2:   (1-alpha)*w_fc2_1+alpha*w_fc2_2,
            b_fc2:   (1-alpha)*b_fc2_1+alpha*b_fc2_2
        } for i in range(55)]
        
        test_dict = {
            x: mnist.test.images,
            y: mnist.test.labels,
            keep_prob: 1.0,
            W_conv1: (1-alpha)*w_conv_1+alpha*w_conv_2,
            b_conv1: (1-alpha)*b_conv_1+alpha*b_conv_2,
            W_fc1:   (1-alpha)*w_fc1_1+alpha*w_fc1_2,
            b_fc1:   (1-alpha)*b_fc1_1+alpha*b_fc1_2,
            W_fc2:   (1-alpha)*w_fc2_1+alpha*w_fc2_2,
            b_fc2:   (1-alpha)*b_fc2_1+alpha*b_fc2_2
        }
        
        train_acc = [accuracy.eval(feed_dict=train_dict[i]) for i in range(55)]
        train_loss = [sess.run(cross_entropy, feed_dict=train_dict[i]) for i in range(55)]
        
        train_acc = np.array(train_acc).mean()
        train_loss = np.array(train_loss).mean()
        
        test_acc = accuracy.eval(feed_dict=test_dict)
        test_loss = sess.run(cross_entropy, feed_dict=test_dict)
        
    return [alpha, train_acc, np.log(train_loss), test_acc, np.log(test_loss)]


# In[7]:


res = []
for i in range(-20, 42):
    res.append(get_acc_loss(i/20))
res = np.array(res)


# In[8]:


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(res[:, 0], res[:, 1], 'b', label='train')
ax1.plot(res[:, 0], res[:, 3], 'b--', label='test')
ax1.set_ylabel('accuracy')
for t in ax1.get_yticklabels():
        t.set_color('b')
ax1.legend()
ax1.set_xlabel('alpha')

ax2.plot(res[:, 0], res[:, 2], 'r', label='train')
ax2.plot(res[:, 0], res[:, 4], 'r--', label='test')
for t in ax2.get_yticklabels():
        t.set_color('r')
ax2.set_ylabel('loss (log scale)')
plt.savefig('3-3_part1.png')
plt.show()

