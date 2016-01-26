import numpy as np

alpha=10.0
n=2
k=2
m=1
W1 = np.matrix(np.random.random((n+1, k))) - 0.5
W2 = np.matrix(np.random.random((k+1, m))) - 0.5
input = np.matrix('0 0; 1 0; 0 1; 1 1')
output = np.matrix('0; 1; 1; 0')

gamma_min = 1e-6
gamma_max = 10
u = 5.0
d = 0.5
gamma_1 = np.zeros_like(W1)
gamma_2 = np.zeros_like(W2)

E = 1

dE1_old = np.zeros(shape=(3,2))
dE2_old = np.zeros(shape=(3,1))

while E > 1e-5:

    dE2 = 0
    dE1 = 0


    E=0
    for i in range(len(input)):

        o = input[i]
        y = output[i]

        o_dach = np.append(o, [[1]], axis=1)

        net1 = o_dach*W1
        o1 = 1.0/(1+np.exp(-net1))

        o1_dach = np.append(o1, [[1]], axis=1)
        net2 = o1_dach*W2
        o2 = 1.0/(1+np.exp(-net2))

        e = (o2-y).T

        f = np.power(e,2)/2.0
        E = E+sum(f)
        # E = sum(f)

        a1 = np.multiply(o1,(1-o1))
        D1 = np.diag(np.array(a1)[0])

        a2 = np.multiply(o2,(1-o2))
        D2 = np.diag(np.array(a2)[0])

        s2 = D2*e
        s1 = D1*W2[:-1, :]*s2

        #delta_W2=(s2 * np.append(o1, [[1]], axis=1)).T
        #delta_W1=(s1 * np.append(o, [[1]], axis=1)).T
        # W1=W1+delta_W1
        # W2=W2+delta_W2

        dE2 += (s2 * np.append(o1, [[1]], axis=1)).T
        dE1 += (s1 * np.append(o, [[1]], axis=1)).T


    mask = np.multiply(dE1, dE1_old)
    gamma_1_old = gamma_1
    gr = mask[mask > 0]
    lo = mask[mask < 0] * gamma_1.reshape(6,1)
    eq = mask[mask == 0] * gamma_1.reshape(6,1)
    ar1 = np.ma.masked_where(mask > 0, mask)
    ar2 = np.minimum(gamma_1_old * u, gamma_max)
    res = np.multiply(ar1, ar2)
    gamma_1 = np.multiply(ar2,ar1) \
              + np.multiply(np.maximum(gamma_1_old * d, gamma_min),np.ma.masked_where(mask < 0, mask)) \
              + np.multiply(gamma_1_old, np.ma.masked_where(mask == 0, mask))

    #if dE1 * dE1_old.T > 0:
    #    gamma_1 = min(gamma_1 * u, gamma_max)
    #elif dE1 * dE1_old.T < 0:
    #    gamma_1 = max(gamma_1 * d, gamma_min)#

    #if dE2 * dE2_old.T > 0:
    #    gamma_2 = min(gamma_2 * u, gamma_max)
    #elif dE2 * dE2_old.T < 0:
    #    gamma_2 = max(gamma_2 * d, gamma_min)

    W1 += np.multiply(- gamma_1, np.sign(dE1))
    W2 += np.multiply(- gamma_2, np.sign(dE2))


    E_old = E

    print E

print W1
print W2


