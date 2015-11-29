import io
import numpy as np
import math
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt
import sklearn.cluster as cl
import scipy.stats as stat

def load_csv(path, char = ',', cast=lambda x: float(x), max_width = 0):
    f = io.open(path, 'r')
    line = f.readline()
    data = []

    while (line):
        values = np.array(line.strip(' \t\n\r').split(char))
        if (max_width == 0):
            max_width = len(values)
        data.append(np.array([cast(x) for x in values[:max_width]]))
        line = f.readline()

    return np.array(data)

def cluster_points(data, centers, norm = lambda x,mu: np.linalg.norm(x-mu)):
    clusters  = []
    for x in data:
        bestcenter = np.argmin([norm(x,mu) for mu in centers], axis=0)
        clusters.append(bestcenter)
    return np.array(clusters)


def cluster_points_pdf(data, centers, covs):
    clusters  = []
    for x in data:
        bestcenter = np.argmin(np.array([stat.multivariate_normal(mean=mu, cov=co).pdf(x) for mu, co in zip(centers, covs)]))
        clusters.append(bestcenter)
    return np.array(clusters)

def reevaluate_centers(data, clusters):
    centers = []
    #cl = np.unique(clusters)
    cl_max = np.max(clusters)
    groups = []
    for cl in range(cl_max + 1):
        map = np.array([c == cl for c in clusters])
        groups.append(np.array(data[map]))

    for group in groups:
        if (len(group) == 0):
            centers.append(np.array([0,0]))
        else:
            centers.append(np.mean(group, axis = 0))
    return np.array(centers)

def reevaluate_centers_pdf(data, clusters):
    centers = []
    covs = []
    #cl = np.unique(clusters)
    cl_max = np.max(clusters)
    groups = []
    for cl in range(cl_max + 1):
        map = np.array([c == cl for c in clusters])
        groups.append(np.array(data[map]))

    for group in groups:
        if (len(group) == 0):
            centers.append(np.array([0,0]))
            covs.append(np.array([[1,0],[0,1]]))
        else:
            centers.append(np.mean(group, axis = 0))
            covs.append(np.cov(group.T))

    return (np.array(centers), np.array(covs))

p = np.array([(1,-1),(2,1),(4,-1),(5,1)])

mu = np.mean(p, 0)

print ("Mittelwerte: \mu_1 = %s, \mu_2 = %s" % (mu[0], mu[1]))


c = np.cov(p)

print ("C = \\begin{pmatrix} %s & %s \\\\ %s & %s \\end{pmatrix}" % (c[0,0], c[0,1], c[1,0], c[1,1]))

cov = np.matrix([[2.5, 0.5], [0.5, 1.0]])

print ("cov %s" % cov)

l1 = 7.0/4.0 + math.sqrt(13.0)/4.0
l2 = 7.0/4.0 - math.sqrt(13.0)/4.0

print ("\\lambda_1 = %0.4f" % l1)
print ("\\lambda_2 = %0.4f" % l2)

v1 = np.array([-2.0*(1.0-l1), 1.0])
v1n = v1 / np.linalg.norm(v1)

v2 = np.array([-2.0*(1.0-l2), 1.0])
v2n = v2 / np.linalg.norm(v2)

print ("\\vec{v_1} = \\begin{pmatrix} %s \\\\ %s \\end{pmatrix}" % (v1n[0], v1n[1]))
print ("\\vec{v_2} = \\begin{pmatrix} %s \\\\ %s \\end{pmatrix}" % (v2n[0], v2n[1]))

n1 = np.array([1.0, 0.0])
n2 = np.array([0.0, 1.0])

T = np.matrix([[ np.dot(v1, n1), np.dot(v2, n1)], [ np.dot(v1, n2), np.dot(v2, n2) ]])

print ("T = \\begin{pmatrix} %0.4f & %0.4f \\\\ %0.4f & %0.4f \\end{pmatrix}" % (T[0,0], T[0,1], T[1,0], T[1,1]))

p1 = np.matrix([1,-1])
p2 = np.matrix([2, 1])
p3 = np.matrix([4,-1])
p4 = np.matrix([5, 1])

p1_t = T * p1.T
p2_t = T * p2.T
p3_t = T * p3.T
p4_t = T * p4.T


print ("\\vec{p_{t1}} = \\begin{pmatrix} %s \\\\ %s \\end{pmatrix}" % (p1_t[0], p1_t[1]))
print ("\\vec{p_{t2}} = \\begin{pmatrix} %s \\\\ %s \\end{pmatrix}" % (p2_t[0], p2_t[1]))
print ("\\vec{p_{t3}} = \\begin{pmatrix} %s \\\\ %s \\end{pmatrix}" % (p3_t[0], p3_t[1]))
print ("\\vec{p_{t4}} = \\begin{pmatrix} %s \\\\ %s \\end{pmatrix}" % (p4_t[0], p4_t[1]))

M = T * cov * T.T

print ("M = \\begin{pmatrix} %0.4f & %0.4f \\\\ %0.4f & %0.4f \\end{pmatrix}" % (M[0,0], M[0,1], M[1,0], M[1,1]))
print ("M: %s" % M)



#np.array([[1,-1), (2,1), (4,-1), (5,1)])




mouse = load_csv('data/mouse.csv')
clean = mouse[:,0:2]
centers = np.array([[7,4], [8,6], [9,4]])
n = 12

'''
plt.figure(1, figsize=(20,20))
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.15, hspace=0.15)

for i in range(n):

    clusters = cluster_points(clean, centers)

    ax = plt.subplot(3,4,(i + 1))
    if i % 4 != 0:
        ax.set_yticklabels(())
    if i < 8:
        ax.set_xticklabels(())
    plt.title('%s. Iteration' % (i+1))
    plt.scatter(np.array(clean[:,0]), np.array(clean[:,1]), c=clusters, cmap=plt.cm.colors.ListedColormap(['#2f4f4f', '#8fbc8f', '#008000']), marker='.', s=2, lw = 0)
    plt.scatter([centers[:,0]], [centers[:,1]], color='red', marker='x', s=20)

    centers = reevaluate_centers(clean, clusters)

plt.show()
'''

plt.figure(1, figsize=(20,20))
plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.15, hspace=0.15)

#centers = clean[np.random.randint(clean.shape[0],size=30),:]

centers = np.array([[7,4], [8,6], [9,4]])
covs = [np.matrix([[1, 0], [0, 1]]),np.matrix([[1, 0], [0, 1]]),np.matrix([[1, 0], [0, 1]])]
print ("centers: %s" % centers)

for i in range(n):
    clusters = cluster_points_pdf(clean, centers, covs)

    ax = plt.subplot(3,4,(i + 1))
    if i % 4 != 0:
        ax.set_yticklabels(())
    if i < 8:
        ax.set_xticklabels(())
    plt.title('%s. Iteration' % (i+1))
    plt.scatter(np.array(clean[:,0]), np.array(clean[:,1]), c=clusters, cmap=plt.cm.colors.ListedColormap(['#2f4f4f', '#8fbc8f', '#008000']), marker='.', s=2, lw = 0)
    plt.scatter([centers[:,0]], [centers[:,1]], color='red', marker='x', s=20)

    centers, covs = reevaluate_centers_pdf(clean, clusters)

plt.show()

print ("Finished")