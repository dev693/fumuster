import io
import numpy as np
import math
import matplotlib.pyplot as plt

def load_csv(path, char = ',', cast=lambda x: float(x), max_width = 0):
    f = io.open(path, 'r')
    line = f.readline()
    data = []

    while (line):
        values = line.strip(' \t\n\r').split(char)
        if (max_width == 0):
            max_width = len(values)
        data.append(np.array([cast(x) for x in values[:max_width]]))
        line = f.readline()

    return data


training = load_csv('data/chickwts_training.csv', ',', lambda x: int(x))
testing = load_csv('data/chickwts_testing.csv', ',', lambda x: int(x))
values = range(0, 550, 1)
p = []
max = 6
priori = []
mu = []
var = []

print("\\begin{tabular}[c]{|c|c|c|}")
print("\\hline")
print("Futterklasse & Erwartungswert ($\\mu$) & Varianz ($\\sigma^2$) & A-Priori\\")
print("\\hline")

run = 'b'

for i in range(max):
    weight = np.array(list(map(lambda x: x[1], filter(lambda x: x[2] == i, training))))
    n = len(weight)
    m = np.sum(weight) / n
    v = np.sum(np.power(np.subtract(m, weight), 2.0)) / n

    k = 1.0 / math.sqrt(2 * math.pi * v)
    w = list(map(lambda x: k * math.exp(-1/2 * ((x - m) ** 2)/v), values))
    prior = float(n) / len(training)

    p.append(w)
    if run == 'a':
        plt.plot(values, w, label=("Futterklasse %s" % i))
    print("%s & %0.2f & %0.2f & %0.2f \\\\" % (i, m, v, prior))
    mu.append(m)
    priori.append(prior)
    var.append(v)
    print("\\hline")

print("\\end{tabular}")

if run == 'a':
    plt.title("Aufgabe 1b - Wahrscheinlichkeitsdichte")
    plt.ylabel('Wahrscheinlichkeit p(x)')
    plt.xlabel('Gewicht (x)')
    plt.legend(loc='upper right', shadow=False, fontsize='x-small')
    plt.show()


for test in testing:
    weight = test[1]

    k = None
    ck = range(max)
    pw = len(list(filter(lambda x: x[1] == weight, testing)))

    pck = [(priori[k] *  p[k][weight]) / pw for k in ck]



