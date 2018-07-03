import matplotlib
matplotlib.use('Agg')
import pickle
import matplotlib.pyplot as plt

with open('reward_history', 'rb') as f:
    r = pickle.load(f)
with open('loss_history', 'rb') as f:
    l = pickle.load(f)

if len(l) != len(r):
    print ('r, l length not same')

e = list(range(len(l)))
plt.plot(e, l)
plt.savefig('loss.png')
plt.close()
plt.plot(e, r)
plt.savefig('reward.png')
plt.close()
