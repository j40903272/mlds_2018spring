
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


loss_hidden1 = np.load('mnist_hidden1_loss_.npy')
loss_hidden2 = np.load('mnist_hidden2_loss_.npy')
loss_hidden4 = np.load('mnist_hidden4_loss_.npy')

acc_hidden1 = np.load('mnist_hidden1_acc_.npy')
acc_hidden2 = np.load('mnist_hidden2_acc_.npy')
acc_hidden4 = np.load('mnist_hidden4_acc_.npy')


# In[3]:


plt.plot(loss_hidden1, label='1 hidden layer')
plt.plot(loss_hidden2, label='2 hidden layers')
plt.plot(loss_hidden4, label='4 hidden layers')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.savefig('1-2_mnist_loss.png')
plt.show()


# In[4]:


plt.plot(acc_hidden1, label='1 hidden layer')
plt.plot(acc_hidden2, label='2 hidden layers')
plt.plot(acc_hidden4, label='4 hidden layers')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.savefig('1-2_mnist_acc.png')
plt.show()

