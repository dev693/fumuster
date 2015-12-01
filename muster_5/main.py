import io
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import scipy.stats as stat


def load_csv(path, cast=lambda x: float(x), max_width = 0):
    f = io.open(path, 'r')
    line = f.readline()
    data = []

    while (line):
        values = np.array(line.split())
        if (max_width == 0):
            max_width = len(values)
        data.append(np.array([cast(x.strip(' \t\n\r')) for x in values[:max_width]]))
        line = f.readline()

    return np.array(data)


fish = load_csv("data/fish.txt", "\s+", lambda x: int(x))

print ("fish: %s" % fish)