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

print("\\begin{tabular}[c]{|c|c|c|}")
print("\\hline")
print("Futterklasse & Erwartungswert ($\\mu$) & Varianz ($\\sigma^2$) \\")
print("\\hline")

for i in range(max):
    weight = np.array(list(map(lambda x: x[1], filter(lambda x: x[2] == i, training))))
    n = len(weight)
    µ = np.sum(weight) / n
    v = np.sum(np.power(np.subtract(µ, weight), 2.0)) / n

    k = 1.0 / math.sqrt(2 * math.pi * v)
    w = list(map(lambda x: k * math.exp(-1/2 * ((x - µ) ** 2)/v), values))

    p.append(w)
    plt.plot(values, w, label=("Futterklasse %s" % i))
    print("%s & %0.2f & %0.2f \\\\" % (i, µ, v))
    print("\\hline")

print("\\end{tabular}")

plt.title("Aufgabe 1b - Wahrscheinlichkeitsdichte")
plt.ylabel('Wahrscheinlichkeit p(x)')
plt.xlabel('Gewicht (x)')
plt.legend(loc='upper right', shadow=False, fontsize='x-small')
plt.show()


for test in testing:
    weight = test[1]
    best = 0
    k = None
    for i in range(max):
        current = p[i][weight]
        if best < current:
            best = current
            k = i
    print("weight: %s, class: %s" % (weight, k))

