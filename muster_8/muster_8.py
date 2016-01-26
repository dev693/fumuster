import io
import random
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_csv(path, char = ',', cast = lambda x: float(x)):
    f = io.open(path, 'r')
    line = f.readline()
    data = []

    while(line):
        values = line.strip(' \t\n\r').split(char)

        data.append(np.array([cast(x) for x in values]))

        line = f.readline()


    return np.array(data)

spiders = load_csv("data/spiders.txt")
beta = [0,0]
alpha = 0.1
pos = np.array(list(filter(lambda x: x[1]  == 1, spiders)))
neg = np.array(list(filter(lambda x: x[1]  == 0, spiders)))
plt.scatter(pos[:,0], pos[:,1], c='green', marker='+', label="Positive Marker")
plt.scatter(neg[:,0], neg[:,1], c='red', marker='+', label="Negative Marker")
xachse = np.arange(-0.5, 1.8, 0.01)


for k in range(1001):
    sum = np.array([0,0], dtype=np.float64)
    for n in range(spiders.shape[0]):
        x = np.array([1.0, spiders[n, 0]])
        y = spiders[n, 1]
        e = math.exp(np.dot(beta,x.T))
        P = e / (1 + e)
        sum += x * float((y - P))
    beta = beta + alpha * sum
    print ("beta %s" % (beta))
    if k == 1 or k == 10 or k == 100 or k == 1000:
        f = np.exp(np.add(xachse * beta[1], beta[0]))
        plt.plot(xachse, f / (1 + f), label=("Iteration k=%s" % k))


print("\\beta_0 = %0.4f \\beta_1 = %0.4f \\x_{1/2} = %0.4f" % (beta[0], beta[1], (-beta[0]/beta[1])))
plt.scatter(-beta[0]/beta[1], 0.5, marker='o', lw = 0, label=("1/2 Wkeit bei x = %0.2f" % (-beta[0]/beta[1])))
plt.xlim([np.min(xachse),np.max(xachse)])
plt.ylim([-0.1,1.8])
plt.legend(loc='upper right', shadow=False, fontsize='x-small')
plt.show()


beta0, beta1 = np.meshgrid(np.arange(-100,100,5), np.arange(-100,100,5))
yi = spiders[:,1]
xi = spiders[:,0]

likelihood = np.vectorize(lambda beta0, beta1: np.sum(yi * (beta0 + beta1 * xi)) - np.sum(np.log(1 + np.exp(beta0 + beta1 * xi))))

l = likelihood(beta0, beta1)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(beta0, beta1, l, color='#0000FF', label="log-likelihood")
ax.scatter(beta[0], beta[1], likelihood(beta[0], beta[1]), marker="o", c="#FF0000", s=20, lw = 0, label=("b0 = %0.2f; b1 = %0.2f" % (beta[0],beta[1])))
ax.legend(loc='upper right', shadow=False, fontsize='x-small')
plt.show()
