import io
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import axes3d
import itertools


def load_csv(path, char = ',', cast = lambda x: float(x), max_width = 0):
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

def getIndex(n):
    indexlist = list(range(10)) #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    return list(itertools.combinations(indexlist, n))

def weinDataChoose(weinData, n):
    indexlist = getIndex(n)
    weinDataList = [] #in weinDataList soll der i-te Eintrag die von getIndex(n)[i] ausgeduennte Liste von weinData sein
    for i in range(len(indexlist)):
        weinDataTMP = np.delete(weinData, indexlist[i], 1)
        weinDataList.append(weinDataTMP)
    return weinDataList


weinDataRaw = load_csv('winequality-red.txt', char = ';', max_width = 100)
quality = np.array(weinDataRaw[:, 11])

len = len(quality)
result = []
for n in range(1,11):
    for c in getIndex(n):
        data = [np.ones(len)]

        for i in c:
            data.append(weinDataRaw[:,i])

        weinData = np.array(data)

        x = np.dot(weinData, weinData.T)
        beta = np.dot(np.dot(np.linalg.inv(x), weinData), quality)

        sum = 0
        for i in range(len):
            q = np.dot(beta, weinData[:,i])
            qual = quality[i]
            sum += (q - qual) ** 2

        result.append([n, sum / len])


result = np.array(result)

plt.plot(result[:,0], result[:,1], '.')
plt.xlabel("Anzahl der Parameter")
plt.ylabel("Normierte Summe der quadratischen Abweichung")
axes = plt.gca()
axes.set_xlim([0,11])
plt.show()

#weinDataRaw = load_csv('winequality-red.txt', char = ';')
#weinData = np.delete(weinDataRaw, 11, 1) #Loescht die Spalte mit den Zielwerten
#weinData = np.c_[np.ones(len(np.array(weinData))), np.array(weinData)] #Fuegt Spalte mit Einsen vorne an



#x = np.dot(np.array(weinData).T, np.array(weinData))
#beta = np.dot(np.dot(np.linalg.inv(x), np.array(weinData).T), np.array(np.array(weinDataRaw)[:, 11]))

#linRegWerte = np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(beta[0], np.multiply(beta[1], np.array(weinData)[:, 1])), np.multiply(beta[2], np.array(weinData)[:, 2])), np.multiply(beta[3], np.array(weinData)[:, 3])), np.multiply(beta[4], np.array(weinData)[:, 4])), np.multiply(beta[5], np.array(weinData)[:, 5])), np.multiply(beta[6], np.array(weinData)[:, 6])), np.multiply(beta[7], np.array(weinData)[:, 7])), np.multiply(beta[8], np.array(weinData)[:, 8])), np.multiply(beta[9], np.array(weinData)[:, 9])), np.multiply(beta[10], np.array(weinData)[:, 10])), np.multiply(beta[11], np.array(weinData)[:, 11]))

#quadFehler = np.sum(np.multiply(np.subtract(linRegWerte, np.array(weinDataRaw)[:, 11]), np.subtract(linRegWerte, np.array(weinDataRaw)[:, 11])))
#print quadFehler/len(linRegWerte)

