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

cov0 = np.cov(c0)
cov1 = np.cov(c1)

print("COV_0 = \\begin{pmatrix} %s & %s \\\\ %s & %s \\end{pmatrix}" % (cov0[0,0], cov0[0,1], cov0[1,0], cov0[1,1]))
print("COV_1 = \\begin{pmatrix} %s & %s \\\\ %s & %s \\end{pmatrix}" % (cov1[0,0], cov1[0,1], cov1[1,0], cov1[1,1]))

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
n = len(c0)


mu0 = float(np.dot(m0, wn.T))
mu1 = float(np.dot(m1, wn.T))
muv0 = (mu0 * wn)
muv1 = (mu1 * wn)

print("\\mu_0 = %s" % mu0)
print("\\mu_1 = %s" % mu1)

#print("\\vec{\\mu_0} = %s" % (mu0 * wn))
#print("\\vec{\\mu_1} = %s" % (mu1 * wn))
print("\\vec{m_{0,p}} = \\begin{pmatrix} %s \\ %s \\end{pmatrix}" % (muv0[0], muv0[1]))
print("\\vec{m_{1,p}} = \\begin{pmatrix} %s \\ %s \\end{pmatrix}" % (muv1[0], muv1[1]))

var0 = np.sum([(float(np.dot(p, wn.T)) - mu0)**2 for p in c0]) / n
var1 = np.sum([(float(np.dot(p, wn.T)) - mu1)**2 for p in c1]) / n

print("\\sigma_{0,p} = %s" % var0)
print("\\sigma_{1,p} = %s" % var1)

print("\\vec{w} = \\begin{pmatrix} %s \\ %s \\end{pmatrix}" % (wn[0], wn[1]))

w0 = ((mu0 + mu1) / 2) * wn

print("\\vec{w_0} = \\begin{pmatrix} %s \\ %s \\end{pmatrix}" % (w0[0], w0[1]))

plt.scatter(x0, y0, color='b', marker='.', label='Punkte des 0 Cluster')
plt.scatter(x1, y1, color='r', marker='.', label='Punkte des 1 Cluster')
plt.plot(l[:,0],l[:,1], color='g', label="Diskriminante")
plt.scatter(muv0[0], muv0[1], color='b', marker='x', s=30, linewidths=1.5, label="erwartungswert des 0 Cluster")
plt.scatter(muv1[0], muv1[1], color='r', marker='x', s=30, linewidths=1.5, label="Erwartungswert des 1 Cluster")
plt.scatter(w0[0], w0[1], color='#00FFFF', marker='x', s=30, linewidths=1.5, label="Schnittpunk der Normalverteilungen")


plt.xlim([-100,250])
plt.ylim([-100,200])
plt.legend(loc='upper left', shadow=False, fontsize='x-small')

plt.scatter(m0[0], m0[1], color='b', marker='x', s=30, linewidths=1.5)
plt.scatter(m1[0], m1[1], color='r', marker='x', s=30, linewidths=1.5)
#plt.plot([w0[0],m0[0]],[w0[1],m0[1]], color='black')
#plt.plot([w1[0],m1[0]],[w1[1],m1[1]], color='black')


plt.show()