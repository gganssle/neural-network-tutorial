import numpy as np

w1 = np.random.randn(1,3)
w2 = np.random.randn(3,1)

w1g = np.zeros((1,3))
w2g = np.zeros((3,1))

def sig(x, forward=True):
    if forward:
        return 1 / (1 + np.exp(-x))
    else: # this is the derivative of sigmoid (for backprop)
        return x * (1. - x)

def nn_forward(inpt, w1, w2):
    # LAYER 1 ################
    l1 = np.matmul(inpt, w1)
    l1_sig = sig(l1)
    # LAYER 2 ################
    l2 = np.matmul(l1_sig, w2)

    return l2, l1_sig, l1

def loss(pred, truth):
    return .5 * np.square(pred - truth)

def nn_backward(inpt, l1, l1_sig, l2, target, learning_rate):
    do = np.zeros(l2.shape[0])
    dh = np.zeros(l1.shape[0])

    # OUTPUT LAYER ##############################
        # calculate error signal for output layer
    do = (l2 - target)

        # update gradients for output layer
    for j in range(w2g.shape[0]):
        w2g[j] = do * l1_sig[j]

        # update weights for output layer
    for j in range(w2g.shape[0]):
        w2[j] -= learning_rate * w2g[j]

    # HIDDEN LAYER ##############################
        # calculate error signal for hidden layer
    for j in range(dh.shape[0]):
        dh[j] = sig(l1_sig[j], forward=False) * do * w2[j]

        # update gradients for hidden layer
    for j in range(w1g.shape[1]):
        w1g[0,j] = dh[j] * inpt

        # update weights for hidden layer
    for j in range(w1g.shape[1]):
        w1[0,j] -= learning_rate * w1g[0,j]

''''''
sample = np.array([1])

l2, l1_sig, l1 = nn_forward(sample, w1, w2)
print(l2)
loss(nn_forward(sample, w1, w2)[0], sample * sample)

print(w1)
print(w2.transpose())

nn_backward(sample, l1, l1_sig, l2, 2, 0.1)

print(w1)
print(w2.transpose())

''''''

num_epochs = 100
learning_rate = 0.1

for i in range(num_epochs):
    sample = np.array([1])
    target = np.array([2])

    l2, l1_sig, l1 = nn_forward(sample, w1, w2)

    nn_backward(sample, l1, l1_sig, l2, target, learning_rate)

    print('prediction = ', l2)
    print('loss: ', loss(l2, target))
    print('w1 = ', w1)
    print('w2 = ', w2.transpose(), '\n')
