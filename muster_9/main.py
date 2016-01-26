import numpy as np

def sig(x, deriv=False):
    if (deriv == True):
        return x * (1-x)
    return 1 / (1 + np.exp(-x))


x = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
y = np.array([[0,1,1,0]]).T

np.random.seed(1)

w1 = np.random.random((3,2))
w2 = np.random.random((2,1))

for i in range(10000):
    l0 = x
    l1 = sig(np.dot(l0,w1))
    l2 = sig(np.dot(l1,w2))

    l2_error = y - l2
    l1_error = l1 - l2

    l1_delta = l1_error * sig(l1, True)

    weight += np.dot(l0.T, l1_delta)

print (l1)
print ("weight: %s" % weight)