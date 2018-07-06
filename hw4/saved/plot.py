import matplotlib
matplotlib.use('Agg')
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('reward_history', 'rb') as f:
    r = pickle.load(f)
with open('loss_history', 'rb') as f:
    l = pickle.load(f)

if len(l) != len(r):
    print ('r, l length not same')

e = list(range(len(l)))
plt.figure(figsize=(20, 15))
plt.plot(e, l)
plt.savefig('loss2.png')
plt.close()


ten = []
tmp = []
for rr in r:
    tmp.append(rr)
    if len(tmp) == 30:
        ten.append(np.mean(tmp))
        tmp = []
        
e = list(range(len(ten)))
e = [i*10 for i in e]

plt.figure(figsize=(10,3))
plt.plot(e, ten, linewidth=0.6)
plt.savefig('reward3.png')
plt.close()
