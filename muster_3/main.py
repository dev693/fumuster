import io
import numpy as np
import math
import random
from scipy.spatial import distance
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


def makeRandomUnitVector():
    random.seed(None)
    #rd = random.randint(1, 4)
    x = random.random()
    y = math.sqrt(1 - x ** 2)
    #if(rd == 1):
    return [x, y]
    #if(rd == 2):
    #    return [-x, y]
    #if(rd == 3):
    #    return [-x, -y]
    #else:
    #    return [x, -y]


def make1000RandomUnitVectors():
    list = []
    for i in range(1000):
        list.append(makeRandomUnitVector())
    return list

def project(x, u):#x auf u projezieren
    return (np.dot(x, u)) * u

def computeMu(sample):
    sumX = np.sum(sample[:, 0])
    sumY = np.sum(sample[:, 1])
    return [sumX / len(sample), sumY / len(sample)]

def T(v):
    return v.T

def project_var(cov, u):
    m = [u[0] * cov[0,0]  + u[1] * cov[1,0], u[0] * cov[0,1]  + u[1] * cov[1,1]]
    return np.dot(m , u)

training = load_csv('data/chickwts_training.csv', ',', lambda x: int(x))
testing = load_csv('data/chickwts_testing.csv', ',', lambda x: int(x))
values = range(0, 550, 1)
p = []
max = 6
priori = []
mu = []
var = []

print("\\begin{tabular}[c]{|c|c|c|c|}")
print("\\hline")
print("Futterklasse & Erwartungswert ($\\mu$) & Varianz ($\\sigma^2$) & A-Priori\\")
print("\\hline")

run = '2'

for i in range(max):
    weight = np.array(list(map(lambda x: x[1], filter(lambda x: x[2] == i, training))))
    n = len(weight)
    m = np.sum(weight) / n
    v = np.sum(np.power(np.subtract(m, weight), 2.0)) / n

    k = 1.0 / math.sqrt(2 * math.pi * v)
    w = list(map(lambda x: k * math.exp(-1/2 * ((x - m) ** 2)/v), values))
    prior = float(n) / len(training)

    p.append(w)
    if run == 'a' or run == 'all':
        plt.plot(values, w, label=("Futterklasse %s" % i))
    print("%s & %0.2f & %0.2f & %0.2f \\\\" % (i, m, v, prior))
    mu.append(m)
    priori.append(prior)
    var.append(v)
    print("\\hline")

print("\\end{tabular}")

if run == '1a' or run == 'all':
    plt.title("Aufgabe 1b - Wahrscheinlichkeitsdichte")
    plt.ylabel('Wahrscheinlichkeit p(x)')
    plt.xlabel('Gewicht (x)')
    plt.legend(loc='upper right', shadow=False, fontsize='x-small')
    #plt.show()

if run == '1b' or run == 'all':

    confusion = np.zeros((max,max))
    right = 0

    for test in testing:
        weight = test[1]
        cl = test[2]

        ck = range(max)
        pw = len(list(filter(lambda x: x[1] == weight, testing)))

        pck = np.array([(priori[k] *  p[k][weight]) / pw for k in ck])
        best = np.argmax(pck)

        if (best == cl):
            right += 1

        confusion[best, cl] += 1



    print ("\\begin{pmatrix}")
    for x in range(max):
        for y in range(max):
            print ("%s" % confusion[x, y], end="")
            if (y != max):
                print (" & ", end="")
        print (" \\\\")
    print ("\\end{pmatrix}")

    print ("\nKlassifikationsgüte: %0.2f %%" % (right / len(testing) * 100))


if run == "2" or run == "2a" or run == "all":
    mu1   = (5, 10)
    cov1  = [[1,1],[1,1]]

    x1,y1 = np.random.multivariate_normal(mu1, cov1, 100).T

    mu2   = (5,5)
    cov2  = [[2,-1],[-1,2]]

    x2,y2 = np.random.multivariate_normal(mu2, cov2, 100).T

    mu3   = (10, 6)
    cov3  = [[0.1, 0],[0, 3]]

    x3,y3 = np.random.multivariate_normal(mu3, cov3, 100).T

    plt.plot(x1, y1, 'b.', label='m = ( 5,10)')
    plt.plot(x2, y2, 'g.', label='m = ( 5, 5)')
    plt.plot(x3, y3, 'r.', label='m = (10, 6)')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.show()

    r_cov1 = np.cov(x1, y1)
    r_cov2 = np.cov(x2, y2)
    r_cov3 = np.cov(x3, y3)
    print ("\\Sigma_1 = \\begin{pmatrix} %0.2f & %0.2f \\\\ %0.2f & %0.2f \\end{pmatrix}" % (r_cov1[0,0], r_cov1[0,1], r_cov1[1,0], r_cov1[1,1]))
    print ("\\Sigma_2 = \\begin{pmatrix} %0.2f & %0.2f \\\\ %0.2f & %0.2f \\end{pmatrix}" % (r_cov2[0,0], r_cov2[0,1], r_cov2[1,0], r_cov2[1,1]))
    print ("\\Sigma_3 = \\begin{pmatrix} %0.2f & %0.2f \\\\ %0.2f & %0.2f \\end{pmatrix}" % (r_cov3[0,0], r_cov3[0,1], r_cov3[1,0], r_cov3[1,1]))

    r1 = np.corrcoef(x1, y1)
    r2 = np.corrcoef(x2, y2)
    r3 = np.corrcoef(x3, y3)

    print ("\\K_{1} = \\begin{pmatrix} %0.2f & %0.2f \\\\ %0.2f & %0.2f \\end{pmatrix}" % (r1[0,0], r1[0,1], r1[1,0], r1[1,1]))
    print ("\\K_{2} = \\begin{pmatrix} %0.2f & %0.2f \\\\ %0.2f & %0.2f \\end{pmatrix}" % (r2[0,0], r2[0,1], r2[1,0], r2[1,1]))
    print ("\\K_{3} = \\begin{pmatrix} %0.2f & %0.2f \\\\ %0.2f & %0.2f \\end{pmatrix}" % (r3[0,0], r3[0,1], r3[1,0], r3[1,1]))


    #x = np.random.uniform(0,100,1000)
    #y = np.random.uniform(0,100,1000)

    #l = np.sqrt(x ** 2 + y ** 2)
    #x = x / l
    #y = y / l

    u = make1000RandomUnitVectors()
    mu1_r = computeMu(np.array(list(zip(x1,y1))))
    mu2_r = computeMu(np.array(list(zip(x2,y3))))
    mu3_r = computeMu(np.array(list(zip(x3,y3))))

    v_project = np.vectorize(project)

    mu_p1 = v_project(mu1_r,u)
    mu_p2 = v_project(mu2_r,u)
    mu_p3 = v_project(mu3_r,u)

    var1 = []
    var2 = []
    var3 = []
    for u_i in u:
        var1.append(project_var(r_cov1, u_i))
        var2.append(project_var(r_cov2, u_i))
        var3.append(project_var(r_cov3, u_i))

    s_u12 = []
    s_u13 = []
    s_u23 = []
    for i in range(1000):
        s_u12.append(distance.euclidean(mu_p1[i],mu_p2[i]) / (var1[i] + var2[i]))
        s_u13.append(distance.euclidean(mu_p1[i],mu_p3[i]) / (var1[i] + var3[i]))
        s_u23.append(distance.euclidean(mu_p2[i],mu_p3[i]) / (var2[i] + var3[i]))

    i12 = np.argmax(s_u12)
    i13 = np.argmax(s_u13)
    i23 = np.argmax(s_u23)



    plt.plot([0, u[i12][0] * 12], [0, u[i12][1] * 12], 'g-', label="s_u12")
    plt.plot([0, u[i13][0] * 12], [0, u[i13][1] * 12], 'b-', label="s_u13")
    plt.plot([0, u[i23][0] * 12], [0, u[i23][1] * 12], 'r-', label="s_u23")
    plt.legend(loc='upper right', shadow=False, fontsize='x-small')
    plt.show()
    #plt.plot(x2, y2, 'g.', label='m = ( 5, 5)')
    #plt.plot(x3, y3, 'r.', label='m = (10, 6)')
    print ("HUHU")




