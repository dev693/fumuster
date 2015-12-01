import io
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d


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

    return data


fishData = load_csv('fish.txt', cast = lambda x: int(x))
fishLength = np.array(fishData)[:, 3]
fish1AgeTemp = []
for i in range(len(fishData)):
    fish1AgeTemp.append([1, fishData[i][1], fishData[i][2]])

x = np.dot(np.array(fish1AgeTemp).T, np.array(fish1AgeTemp))
beta = np.dot(np.dot(np.linalg.inv(x), np.array(fish1AgeTemp).T), np.array(fishLength))
print (beta)

for i in range(len(fishData)):
    print ('geschaetzt: %i exakt: %i' % ((beta[0] + beta[1] * fish1AgeTemp[i][1] + beta[2] * fish1AgeTemp[i][2], fishLength[i])))

xs = np.array(fish1AgeTemp)[:, 1]
ys = np.array(fish1AgeTemp)[:, 2]
zs = (np.add(np.add(beta[0], np.multiply(beta[1], xs)), np.multiply(beta[2], ys)))


xx, yy = np.meshgrid(xs, ys);
zz = beta[0] + np.multiply(beta[1], xx) + np.multiply(beta[2], yy)

#print ("x: %s" % xs)
#print ("y: %s" % ys)
#print ("z: %s" % zs)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Alter')
ax.set_ylabel('Temperatur')
ax.set_zlabel('Laenge')
ax.set_zlim([0,6000])
#ax.set_xlim([0,160])
#ax.set_ylim([24,33])
#ax.plot_wireframe(xs, ys, zs, color='green')
#ax.plot_wireframe(xx, yy, zz, color='green', label="Regressionsebene")


ax.scatter(xs, ys, fishLength, color = '#00FF00', marker = '.', label="Korrekte Länge") #exakt
ax.scatter(xs, ys, zs, color = '#0000FF', marker = '.', label="Schätzung durch Regression") #lineare Regression
ax.scatter(xs, ys, np.absolute(fishLength - zs), color='#FF0000', marker = '.', label="Fehler")

ax.legend(loc='upper right', shadow=False, fontsize='x-small')

plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Alter')
ax.set_ylabel('Temperatur')
ax.set_zlabel('Laenge')

ax.scatter(xs, ys, fishLength, color = '#00FF00', marker = '.', label="Korrekte Länge") #exakt
ax.plot_wireframe(xx, yy, zz, color='#0000FF', label="Regressionsebene")
ax.legend(loc='upper right', shadow=False, fontsize='x-small')

plt.show()
#ax.plot_wireframe(xx, yy, zz, color='green', label="Regressionsebene")
