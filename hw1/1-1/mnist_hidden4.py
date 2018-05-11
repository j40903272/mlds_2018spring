
# coding: utf-8

# In[1]:


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np


# In[2]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[3]:


def weight_variable(shape):
  init = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(init)

def bias_variable(shape):
  init = tf.constant(0.1, shape=shape)
  return tf.Variable(init)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 


# In[4]:


x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([9, 9, 1, 1])
b_conv1 = bias_variable([1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_fc1 = weight_variable([14*14*1, 13])
b_fc1 = bias_variable([13])
h_pool1_flat = tf.reshape(h_pool1, [-1, 14*14*1])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([13, 11])
b_fc2 = bias_variable([11])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([11, 11])
b_fc3 = bias_variable([11])
h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

W_fc4 = weight_variable([11, 10])
b_fc4 = bias_variable([10])
h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)

W_fc5 = weight_variable([10, 10])
b_fc5 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc4_drop, W_fc5) + b_fc5)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print(total_parameters)


# In[5]:


sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

loss = []
acc = []

for i in range(55*500):
    batch = mnist.train.next_batch(1000)
    train_step.run(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
    # an epoch = 55 iterations
    if i % 55 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
        acc.append(train_accuracy)
        train_loss = sess.run(cross_entropy, feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
        loss.append(train_loss)
        

print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))


# In[6]:


total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print(total_parameters)


# In[6]:


loss = np.array(loss)
acc = np.array(acc)
np.save('cnn_hidden4_loss_.npy', loss)
np.save('cnn_hidden4_acc_.npy', acc)

