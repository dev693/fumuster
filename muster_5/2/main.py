import io
import numpy as np
import matplotlib.pyplot as plt
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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = np.array(fish1AgeTemp)[:, 1]
ys = np.array(fish1AgeTemp)[:, 2]
ax.set_xlabel('Alter')
ax.set_ylabel('Temperatur')
ax.set_zlabel('Laenge')

ax.scatter(xs, ys, fishLength, color = 'r', marker = 'o') #exakt
ax.scatter(xs, ys, (np.add(np.add(beta[0], np.multiply(beta[1], np.array(fish1AgeTemp)[:, 1])), np.multiply(beta[2], np.array(fish1AgeTemp)[:, 2]))), color = 'b', marker = 'o') #lineare Regression

ax.plot_surface(np.array(fish1AgeTemp)[:, 1], np.array(fish1AgeTemp)[:, 2], np.add(np.add(beta[0], np.multiply(beta[1], np.array(fish1AgeTemp)[:, 1])), np.multiply(beta[2], np.array(fish1AgeTemp)[:, 2])), cmap='summer')
plt.show()



