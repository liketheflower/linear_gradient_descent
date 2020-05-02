# simulate the l1 loss and l2 loss
# simulate w*x = y where x is a fixed number 2
# It means we are going to simulate 2*w = y
# a noise item will be added which is z. 
# 2*w + z = y
# if z is 0, then there is no noise, 
# else some gaussian noise will be added
import numpy as np
import random
# number of samples
N = 200
w = 20
# our target w is 1.0

x = 2*np.ones((N,))
y = 1.0*x

def loss(y_hat, y, loss='L1'):
    if loss=='L1':
        return  np.sum(y_hat - y)
    elif loss == 'L2':
        return np.sum(np.square(y_hat-y))


epoches = 10
batch_size = 4
batch_start_idx = list(range(0, N, batch_size))

l1_losses, l2_losses = [], []
w_l1_loss, w_l2_loss = [], []
# set learning rate as 1.0
lr = 0.00001

for loss_type in ['L1', 'L2']:
    w = 20
    for epoch in range(epoches):
        for batch_i in batch_start_idx:
            this_x, this_y = x[batch_i*batch_size:(batch_i+1)*batch_size], y[batch_i*batch_size:(batch_i+1)*batch_size]
            y_hat = w*this_x
            L = loss(y_hat, this_y, loss_type)
            if loss_type == 'L1':
                l1_losses.append(L)
                print(f"{loss_type}, {L}")
                # L = y_hat -y , dL/dy_hat = 1
                # dy_hat/dw = x
                # dL/dw = Loss *(1)*(x)
                gradient = np.sum(L*x)
                w -= lr*gradient
                w_l1_loss.append(w)
            elif loss_type == 'L2':
                l2_losses.append(L)
                print(f"{loss_type}, {L}")
                # L = (y_hat -y)**2 , dL/dy_hat = 2(y_hat - y)
                # dy_hat/dw = x
                # dL/dw = Loss *(2*(y_hat-y))*(x)
                gradient = np.sum(L*2*(y_hat-this_y)*this_x)
                w -= lr*gradient
                w_l2_loss.append(w)
print(l1_losses)
print(l2_losses)
print(w_l1_loss)
print(w_l2_loss)
