
# coding: utf-8

# In[1]:


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# In[3]:


def weight_variable(shape, name):
  init = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(init, name=name)

def bias_variable(shape, name):
  init = tf.constant(0.1, shape=shape)
  return tf.Variable(init, name=name)
  
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 


# In[4]:


x = tf.placeholder(tf.float32, [None, 784], name='x')
y = tf.placeholder(tf.float32, [None, 10], name='y')
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = weight_variable([5, 5, 1, 32], 'w_conv1')
b_conv1 = bias_variable([1], 'b_conv1')
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_fc1 = weight_variable([14*14*32, 64], 'w_fc1')
b_fc1 = bias_variable([64], 'b_fc1')
h_pool1_flat = tf.reshape(h_pool1, [-1, 14*14*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([64, 10], 'w_fc2')
b_fc2 = bias_variable([10], 'b_fc2')

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2, name='y_conv')

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_conv), reduction_indices=[1]), name='cross_entropy')
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[5]:


def test_model(path):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, path)
        
        train_dict = [{
            x: mnist.train.images[1000*i:1000*(i+1)],
            y: mnist.train.labels[1000*i:1000*(i+1)],
            keep_prob: 1.0,
        } for i in range(55)]
        
        test_dict = {
            x: mnist.test.images,
            y: mnist.test.labels,
            keep_prob: 1.0,
        }
        
        train_acc = [accuracy.eval(feed_dict=train_dict[i]) for i in range(55)]
        train_loss = [sess.run(cross_entropy, feed_dict=train_dict[i]) for i in range(55)]
        
        train_acc = np.array(train_acc).mean()
        train_loss = np.array(train_loss).mean()
        
        test_acc = accuracy.eval(feed_dict=test_dict)
        test_loss = sess.run(cross_entropy, feed_dict=test_dict)
    
        grad = tf.gradients(cross_entropy, x)[0].eval(feed_dict={x: mnist.train.images[:1000], y: mnist.train.labels[:1000], keep_prob: 1.0})
        sensitivity = np.linalg.norm(grad, 'fro')
    
    return [train_acc, np.log(train_loss), test_acc, np.log(test_loss), sensitivity]


# In[6]:


res = [test_model('model_32/batch_32.chkp'),
       test_model('model_64/batch_64.chkp'),
       test_model('model_128/batch_128.chkp'),
       test_model('model_512/batch_512.chkp'),
       test_model('model_1024/batch_1024.chkp'),
       test_model('model_2048/batch_2048.chkp'),
       test_model('model_4096/batch_4096.chkp')]
res = np.array(res)


# In[7]:


xaxis = [32, 64, 128, 512, 1024, 2048, 4096]
xaxis = (np.array(xaxis))

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.plot(xaxis, res[:, 0], 'b', label='train')
ax1.plot(xaxis, res[:, 2], 'b--', label='test')
ax1.set_ylabel('accuracy')
ax1.set_xscale('log')
for t in ax1.get_yticklabels():
        t.set_color('b')
ax1.legend()
ax1.set_xlabel('batch size (log scale)')

ax2.plot(xaxis, res[:, 1], 'r', label='train')
ax2.plot(xaxis, res[:, 3], 'r--', label='test')
for t in ax2.get_yticklabels():
        t.set_color('r')
ax2.set_ylabel('loss (log scale)')

plt.savefig('3-3_part2_loss_acc.png')
plt.show()


# In[8]:


plt.plot(xaxis, res[:, 4], 'r')
plt.xscale('log')
plt.xlabel('batch size (log scale)')
plt.ylabel('sensitivity')
plt.savefig('3-3_part2_sensitivity.png')
plt.show()

