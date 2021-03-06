import numpy as np
import random
import matplotlib.pyplot as plt
N = 100
x = np.random.randn(N)
w, b = 2, 3

y = w*x + b 
z = np.random.normal(0, 0.1, N)
y = y+z
W, b = np.random.randn(), np.random.randn()

epoches = 30
batch_size = 4

batch_idx = list(range(N//batch_size+1))
lr = 0.0002
batch_losses = []
epoch_losses = []
for epoch in range(epoches):
    random.shuffle(batch_idx)
    epoch_loss = 0
    for i in  batch_idx:
        this_batch_x = x[i*batch_size:(i+1)*batch_size]
        this_batch_y = y[i*batch_size:(i+1)*batch_size]
        if this_batch_x.size == 0:continue
        y_hat = this_batch_x*w + b
        LOSS = np.sum(np.square(this_batch_y-y_hat))
        epoch_loss += LOSS
        batch_losses.append(LOSS)
        dL_dw = np.sum(2*this_batch_x*(y_hat-this_batch_y))
        dL_db = np.sum(2*(y_hat-this_batch_y))
        w -= lr*dL_dw
        b -= lr*dL_db
    print(epoch_loss)
    epoch_losses.append(epoch_loss)
#print(epoch_losses)
plt.plot(range(len(batch_losses)), batch_losses)
plt.savefig('loss.png')
plt.show()
print(w, b)



        
