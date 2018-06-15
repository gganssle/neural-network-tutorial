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
