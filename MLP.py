import numpy as np
import random
import matplotlib.pyplot as plt

def np_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))


N = 1000
# features
M = 10
x = np.random.randn(N, M)
y = np.random.randint(10, size=N)
def one_hot(a, number_category=10):
    res = [0]*number_category
    assert 0<=a<number_category
    res[int(a)] = 1
    return res
y = [one_hot(a) for a in y.tolist()]
y = np.array(y, dtype=float)
print(y.shape)

# two layers
w0 = np.random.normal(0, 0.1, size=(10, 16))
b0 = np.zeros(16)

dw0 =np.zeros_like(w0)
db0 = np.zeros_like(b0)

w1 = np.random.normal(0, 0.1, size=(16, 10))
b1 = np.zeros(10)

dw1 =np.zeros_like(w1)
db1 = np.zeros_like(b1)





epoches = 100
batch_size = 64

begins = np.arange(0, N, batch_size)
lr = 0.001
batch_losses = []
epoch_losses = []

class Model():
    def __init__(self,
                 lr, 
                 w0,
                 dw0,
                 b0,
                 db0,
                 w1,
                 dw1,
                 b1,
                 db1):
        self.lr = lr
        self.w0 = w0
        self.dw0 = dw0
        self.b0 = b0
        self.db0 = db0
        self.w1 = w1
        self.dw1 = dw1
        self.b1 = b1
        self.db1 = db1
    def zero_grad(self):
        for grad in self.grads:
            grad = 0.0


    def forward(self, x0):
        self.x0 = x0
        self.y0 = self.x0@self.w0 + self.b0
        self.tanh = np.tanh(self.y0)
        self.y_hat = self.tanh@self.w1 + self.b1
        return self.y_hat

    def backward(self, y):
        #(y_hat-y)*(y_hat-y)=L
        self.dy_hat = self.y_hat-y
        # w1@y0+b1 = y_hat
        self.dw1 = self.tanh.T@self.dy_hat
        self.db1 = np.sum(self.dy_hat, axis=0)
        self.dtanh = self.dy_hat @ self.w1.T
        # w0@x0+b0 = y0
        self.dy0 = self.dtanh*(1-self.y0*self.y0)
        #print("self dy0 shape",self.dy0.shape)
        self.dw0 = self.x0.T @ self.dy0
        self.db0 = np.sum(self.dy0, axis=0)

    def step(self):
        self.parameters =[self.w0, self.b0, self.w1, self.b1]
        self.grads =  [self.dw0, self.db0, self.dw1, self.db1]
        for parameter, grad in zip(self.parameters, self.grads):
            parameter -= lr*grad
lr = 0.0001
model = Model(lr=lr, 
              w0=w0, 
              dw0=dw0,
              b0=b0,
              db0=db0,
              w1=w1,
              dw1=dw1,
              b1=b1,
              db1=db1)

for epoch in range(epoches):
    random.shuffle(begins)
    epoch_loss = 0
    for begin in  begins:
        end = begin + batch_size
        this_x, this_y = x[begin:end], y[begin:end]

        y_hat = model.forward(this_x)
        LOSS = np.sum(np.square(this_y-y_hat))
        epoch_loss += LOSS
        batch_losses.append(LOSS)
        model.backward(this_y)
        model.step()
        model.zero_grad()
    print(epoch_loss)
    epoch_losses.append(epoch_loss)
#print(epoch_losses)
plt.plot(range(len(epoch_losses)), epoch_losses)
plt.savefig('mlp_loss.png')
plt.show()



        
