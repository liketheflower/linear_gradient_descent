# simulate the abs_L1 loss and l2 loss
# simulate w*x = y where x is a fixed number 2
# It means we are going to simulate 2*w = y
# a noise item will be added which is z. 
# 2*w + z = y
# if z is 0, then there is no noise, 
# else some gaussian noise will be added
import numpy as np
import random
import matplotlib.pyplot as plt
# number of samples
N = 20
INIT_w = 20
# our target w is 1.0
sigma = 5
add_noise = True
x = 2*np.ones((N,))
if add_noise:
    z = np.random.normal(0, sigma, N)
    y = 1.0*x + z
else:
    y = 1.0*x
def loss(y_hat, y, loss='abs_L1'):
    if loss=='abs_L1':
        # print('abs_L1 loss')
        return  np.sum(y_hat - y)
    elif loss == 'L2':
        #print('L2 loss')
        return np.sum(np.square(y_hat-y))

#print(x)
#print(y)
epoches = 2000
batch_size = 4
batch_start_idx = list(range(0, N, batch_size))

abs_L1_losses, l2_losses = [], []
w_abs_L1_loss, w_l2_loss = [], []
# set learning rate as 1.0
init_lr = 0.01
decay = 1/2000
for loss_type in ['abs_L1', 'L2']:
    w = INIT_w 
    iterations = 1
    lr = init_lr 
    lr_s = []
    for epoch in range(epoches):
        for batch_i in batch_start_idx:
            iterations += 1
            lr = init_lr*(1. / (1. + decay * iterations))
            if iterations==2000:print(f"lr {lr}, iterations {iterations}, decay {decay}")
            lr_s.append(lr)
            this_x, this_y = x[batch_i:batch_i+batch_size], y[batch_i:batch_i+batch_size]
            y_hat = w*this_x
            L = loss(y_hat, this_y, loss_type)
            if loss_type == 'abs_L1':
                abs_L1_losses.append(abs(L))
                # print(f"{loss_type}, {L}")
                # L = y_hat -y , dL/dy_hat = 1
                # dy_hat/dw = x
                # dL/dw = Loss *(1)*(x)
                gradient = np.sum(x)
                #print(gradient)
                if L<0:gradient = -gradient
                w -= lr*gradient
                w_abs_L1_loss.append(w)
            elif loss_type == 'L2':
                l2_losses.append(L)
                #print(f"{loss_type}, {L}")
                # L = (y_hat -y)**2 , dL/dy_hat = 2(y_hat - y)
                # dy_hat/dw = x
                # dL/dw = Loss *(2*(y_hat-y))*(x)
                gradient = np.sum(2*(y_hat-this_y)*this_x)
                w -= lr*gradient
                w_l2_loss.append(w)
    plt.plot(range(len(lr_s)), lr_s, c = 'r')
    plt.xlabel('iteration')
    plt.ylabel('learning rate')
    plt.savefig('learning_rate.png')
    plt.show()
    plt.close()
#print(abs_L1_losses)
#print(l2_losses)
#print(w_abs_L1_loss)
#print(w_l2_loss)
plt.plot(range(len(abs_L1_losses)), abs_L1_losses, c = 'r', label='abs_L1 loss')
plt.plot(range(len(l2_losses)), l2_losses, c = 'g', label='L2 loss')
plt.xlabel('step')
plt.ylabel('loss')
if add_noise:
    plt.title('initialize w:'+str(INIT_w)+' with noise')
else:
    plt.title('initialize w:'+str(INIT_w)+' without noise')
plt.legend(loc='best')
if add_noise:
    plt.savefig('loss_abs_L1_l2_with_noise with sigma = '+str(sigma)+'.png')
else:
    plt.savefig('loss_abs_L1_l2_without_noise.png')
plt.show()
plt.close()
plt.plot(range(len(w_abs_L1_loss)), w_abs_L1_loss, c = 'r', label='abs_L1 loss')
plt.plot(range(len(w_l2_loss)), w_l2_loss, c = 'g', label='L2 loss')
plt.xlabel('step')
plt.ylabel('estimate w (target is 1)')
if add_noise:
    plt.title('initialize w:'+str(INIT_w)+' with noise')
else:
    plt.title('initialize w:'+str(INIT_w)+' without noise')
plt.legend(loc='best')
if add_noise:
    plt.savefig('w_abs_L1_l2_with_noise with sigma = '+str(sigma)+'.png')
else:
    plt.savefig('w_abs_L1_l2_without_noise.png')
plt.show()
