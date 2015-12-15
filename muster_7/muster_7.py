import io
import random
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path, char = ',', cast = lambda x: float(x)):
    f = io.open(path, 'r')
    line = f.readline()
    data = []

    while(line):
        values = line.strip(' \t\n\r').split(char)

        data.append(np.array([cast(x) for x in values]))

        line = f.readline()

    return np.array(data)

def plot_w(w, count = -1, color = None):
    x = w[0]
    y = w[1]
    len = np.linalg.norm(w)
    wn = w / len
    xn = wn[0]
    yn = wn[1]
    n = len * np.array([-yn,xn])
    ww = np.array(list([[0, 0], w]))
    nn = np.array(list([[0, 0], n]))
    if color == None:
        color = '#00%02x00' % ((50 + count * 30) % 256)

    print("w%s: (%s,%s)" % (count, x, y))
    plt.plot(ww[:,0], ww[:,1], c='green')
    plt.plot(nn[:,0], nn[:,1], c='yellow')

def perzeptron(data, omega = 1, plot_step = False):
    o = data[:,2]
    x = data[:,0:2]
    w = np.array([0,0])
    changes = 0
    nochanges = 0

    for k in range(100):
        #for i in range(x.shape[0]):
        i =  random.randint(0,x.shape[0]-1)
        xi = x[i]
        op = (np.dot(xi.T, w) >= omega)
        ok = (o[i] >= omega)
        if op != ok:
            w = (w + xi if ok else w - xi)
            changes += 1
            nochange = 0
            if plot_step:
                plot_w(w, changes)
        else:
            nochanges += 1

        if nochanges >= 10:
            return w;
        #if count == 0:
            #break

    return w



data = load_csv("data/klausur.txt", char=";")
data = np.array([data[:,0], np.ones(len(data)),data[:,1]]).T
omega = 1

#w = perzeptron(data, omega, True)

#print("changes  : %s" % changes)
#print("nochanges: since %s" % nochange)
#print("w: %s" % w)
pos = np.array(list(filter(lambda x: x[2] >= omega, data)))
neg = np.array(list(filter(lambda x: x[2]  < omega, data)))


#plt.scatter(pos[:,0], pos[:,1], c='green', marker='+')
#plt.scatter(neg[:,0], neg[:,1], c='red', marker='+')
#plt.show()

ws = np.array([perzeptron(data) for i in range(100)])
wmean = ws.mean(axis=0)
plot_w(wmean)
print("w_mean: %s" % wmean)

plt.scatter(pos[:,0], pos[:,1], c='green', marker='+')
plt.scatter(neg[:,0], neg[:,1], c='red', marker='+')
plt.show()
