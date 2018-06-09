import numpy as np
import matplotlib.pyplot as plt

inpt = np.array(range(-9,10))

print(inpt)

def sq(inpt):
    return inpt * inpt

print(sq(inpt))

def der(inpt):
    return 2 * inpt

print(der(inpt))

x = 5
lr = 0.01
hist = []

for i in range(200):
    y = sq(x)
    #print(x, '\t', y, '\t', grad[int(x + 10)])

    x = x - lr * der(x)

    hist.append(x)

plt.figure(figsize=(10,5))
plt.plot(hist)
plt.show()

######################################################################################
######################################################################################

def loss(pred, truth):
    return .5 * np.square(pred - truth)

w1 = np.random.randn(1,3)
w2 = np.random.randn(3,1)

w1g = np.zeros((1,3))
w2g = np.zeros((3,1))

def nn(inpt, w1, w2):
    l1 = np.matmul(inpt, w1)
    l2 = np.matmul(l1, w2)
    return l2, l1

def g2(pred, truth):
    return (pred - truth) * pred
def g1():

'''
sq(sample)
nn(sample, w1, w2)
loss(nn(sample, w1, w2), sq(sample))

sample = np.array([1])
print(loss(nn(sample, w1, w2), sq(sample)))
'''

num_epochs = 10
lr = 0.001

for i in range(num_epochs):
    for j in inpt:
        sample = np.array([j])
        truth  = sq(sample)
        pred   = nn(sample, w1, w2)

        local_loss = loss(pred, truth)
