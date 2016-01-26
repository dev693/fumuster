import numpy as np

def gamma(old, d, u, min, max, mask):
    map = np.zeros_like(mask)
    map[mask > 0] = 1
    gamma = np.multiply(np.minimum(old * u, max), map)

    map = np.zeros_like(mask)
    map[mask < 0] = 1
    gamma += np.multiply(np.maximum(old * d, min), map)

    map = np.zeros_like(mask)
    map[mask == 0] = 1
    gamma += np.multiply(old, map)
    return gamma

if __name__ == '__main__':

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

    dE1_old = np.zeros(shape=(n+1,k))
    dE2_old = np.zeros(shape=(k+1,m))

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


        mask1 = np.multiply(dE1, dE1_old)
        mask2 = np.multiply(dE2, dE2_old)

        gamma_1_old = gamma_1
        gamma_2_old = gamma_2

        gamma_1 = gamma(gamma_1_old, d, u, gamma_min, gamma_max, mask1)
        gamma_2 = gamma(gamma_2_old, d, u, gamma_min, gamma_max, mask2)

        W1 += np.multiply(- gamma_1, np.sign(dE1))
        W2 += np.multiply(- gamma_2, np.sign(dE2))

        dE1_old = dE1
        dE2_old = dE2

        print E

    print W1
    print W2
