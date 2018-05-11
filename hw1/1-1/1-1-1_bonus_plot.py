
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


model1_loss = np.load('model1_loss.npy')
model2_loss = np.load('model2_loss.npy')
model3_loss = np.load('model3_loss.npy')

model1_pred = np.load('model1_pred.npy')
model2_pred = np.load('model2_pred.npy')
model3_pred = np.load('model3_pred.npy')


# In[19]:


plt.plot(model1_loss, label='model1')
plt.plot(model2_loss, label='model2')
plt.plot(model3_loss, label='model3')
plt.yscale('log')
plt.legend(loc='upper right')
plt.ylabel('loss (log scale)')
plt.xlabel('epoch')
plt.savefig('1-1-1_loss')
plt.show()


# In[20]:


x = np.linspace(0, 5, 1000)
y = [ np.cos(i)*i for i in x ]
plt.plot(x, y, 'r', label='cos(x)*x')
plt.plot(x, model1_pred, 'k', label='model1')
plt.plot(x, model2_pred, 'y', label='model2')
plt.plot(x, model3_pred, 'b', label='model3')
plt.legend(loc='upper right')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('1-1-1_function')
plt.show()

