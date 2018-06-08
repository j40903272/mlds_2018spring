
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


loss_hidden1 = np.load('cifar_1_loss.npy')
loss_hidden2 = np.load('cifar_2_loss.npy')
loss_hidden4 = np.load('cifar_4_loss.npy')

acc_hidden1 = np.load('cifar_1_acc.npy')
acc_hidden2 = np.load('cifar_2_acc.npy')
acc_hidden4 = np.load('cifar_4_acc.npy')


# In[3]:


plt.plot(loss_hidden1, label='1 hidden layer')
plt.plot(loss_hidden2, label='2 hidden layers')
plt.plot(loss_hidden4, label='4 hidden layers')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.savefig('1-2_cifar10_loss.png')
plt.show()


# In[4]:


plt.plot(acc_hidden1, label='1 hidden layer')
plt.plot(acc_hidden2, label='2 hidden layers')
plt.plot(acc_hidden4, label='4 hidden layers')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='lower right')
plt.savefig('1-2_cifar10_acc.png')
plt.show()

