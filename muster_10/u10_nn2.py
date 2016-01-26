import numpy as np

alpha=1.
n=16
k=16
m=10
W1 = np.matrix(np.random.random((n+1, k)))-0.5
W2 = np.matrix(np.random.random((k+1, m)))-0.5

# W1 = np.matrix(np.zeros((n+1, k)))+0.5
# W2 = np.matrix(np.zeros((k+1, m)))+0.5

data = np.loadtxt('pendigits-training.txt', unpack=True)

input = np.matrix(data[:-1,:]).T
output = np.matrix(data[-1,:]).T

E=1

while E > 1e-5:

    delta_W2 = np.zeros((k+1,m))
    delta_W1 = np.zeros((n+1,k))

    E=0
    for i in range(len(input)):

        o=input[i]

        y = np.matrix(np.zeros((1,m)))
        y[0, int(output[i])] = 1

        net1 = np.append(o, [[1]], axis=1)*W1
        o1 = 1.0/(1+np.exp(-net1))

        net2 = np.append(o1, [[1]], axis=1)*W2
        o2 = 1.0/(1+np.exp(-net2))

        e = (o2-y).T

        f = np.power(e,2)/2.0
        E = E+sum(f)
# %        E = sum(f)

        a1 = np.multiply(o1,(1-o1))
        D1 = np.diag(np.array(a1)[0])

        a2 = np.multiply(o2,(1-o2))
        D2 = np.diag(np.array(a2)[0])

        s2 = D2*e
        s1=D1*W2[:-1, :]*s2

        delta_W2=(-alpha*s2 * np.append(o1, [[1]], axis=1)).T
        delta_W1=(-alpha*s1 * np.append(o, [[1]], axis=1)).T
        W1=W1+delta_W1
        W2=W2+delta_W2

# %        delta_W2=delta_W2+(-alpha*s2*[o1 1])';
# %        delta_W1=delta_W1+(-1*s1*[o 1])';


# %    W1=W1+delta_W1;
# %    W2=W2+delta_W2;

    print E
    # print W1
    # print W2

print W1
print W2


