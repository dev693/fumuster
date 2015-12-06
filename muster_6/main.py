import io
import numpy as np
import matplotlib.pyplot as plt
import math

def load_csv(path, char = ',', cast = lambda x: float(x)):
    max_width = 0
    f = io.open(path, 'r')
    line = f.readline()
    data = []

    while(line):
        values = line.strip(' \t\n\r').split(char)
        if (max_width == 0):
            max_width = len(values)
        data.append(np.array([cast(x) for x in values[:max_width]]))
        line = f.readline()

    return np.array(data)



data = load_csv('data/fisher.txt')

cluster0 = np.array(list(filter(lambda d: d[2] == 0, data)))
cluster1 = np.array(list(filter(lambda d: d[2] == 1, data)))


print ("cluster 1: %s" % cluster0)
print ("cluster 2: %s" % cluster1)

x0 = cluster0[:,0]
y0 = cluster0[:,1]
c0 = np.array([x0,y0]).T


x1 = cluster1[:,0]
y1 = cluster1[:,1]
c1 = np.array([x1,y1]).T

m0 = np.mean(c0, 0)
m1 = np.mean(c1, 0)

SB = (m1 - m0) * (m1 - m0).T
SW = [np.sum([(p0 - m0)**2 for p0 in c0], 0),np.sum([(p1 - m1)**2 for p1 in c1], 0)]

WS = np.linalg.inv(np.matrix(SW))

w = (np.asarray((WS * np.matrix(m1 - m0).T).T))[0]
# normalize
wn = np.asarray((w.T / np.linalg.norm(w.T)))
m0n = m0 / np.linalg.norm(m0)
m1n = m1 / np.linalg.norm(m1)

s = 200
l = np.array([wn*(-s), wn*s])


w0 = np.array(float(np.dot(m0, wn.T)) * wn)
w1 = np.array(float(np.dot(m1, wn.T)) * wn)

l0 = w0 - m0
l1 = w1 - m1

a0 = float(np.dot(l0, wn.T))
a1 = float(np.dot(l1, wn.T))


plt.scatter(x0, y0, color='b', marker='.', label='0 Cluster')
plt.scatter(x1, y1, color='r', marker='.', label='1 Cluster')
plt.plot(l[:,0],l[:,1], color='g', label="Diskriminante")
plt.scatter(w0[0], w0[1], color='b', marker='x', s=60, linewidths=3, label="mu0 des 0 Cluster")
plt.scatter(w1[0], w1[1], color='r', marker='x', s=60, linewidths=3, label="mu1 des 1 Cluster")


plt.xlim([-100,250])
plt.ylim([-100,200])
plt.legend(loc='upper left', shadow=False, fontsize='x-small')

plt.scatter(m0[0], m0[1], color='b', marker='x', s=60, linewidths=3)
plt.scatter(m1[0], m1[1], color='r', marker='x', s=60, linewidths=3)
#plt.plot([w0[0],m0[0]],[w0[1],m0[1]], color='black')
#plt.plot([w1[0],m1[0]],[w1[1],m1[1]], color='black')


plt.show()