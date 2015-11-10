from tkinter import *
import os
import random
import numpy as np
import math
from PIL import Image, ImageTk
from array import array

def reprint():
    CURSOR_UP_ONE = '\x1b[1A'
    ERASE_LINE = '\x1b[1M'#'\x1b[2K' #'\x1b[1M'
    print('{}\r'.format(x), end="", flush=True)

def load_labels(file, progress = None, max = 0):
    f = open(os.path.dirname(__file__) + file, "rb")
    magic = int.from_bytes(f.read(4), byteorder='big')
    if (magic != 2049):
        raise ValueError("magic number incorrect, expected 2049 found %s" % magic)

    count = int.from_bytes(f.read(4),  byteorder='big')
    if (progress is not None):
        progress(0, count)


    data = []
    index = 0
    bytes = f.read();
    for byte in bytes:
        data.append(byte);
        index += 1
        if (progress is not None):
            progress(index, count)
        if (max != 0 and index >= max):
            break

    if count != index:
        print("Warning: file broken... counts doesn't match (given count: %s in data: %s)" % (count, index))

    return np.array(data)

def load_images(file, progress = None, max = 0):
    f = open(os.path.dirname(__file__) + file, "rb")
    magic = int.from_bytes(f.read(4), byteorder='big')
    if (magic != 2051):
        print("magic number incorrect")
        f.close()
        return []
    count = int.from_bytes(f.read(4),  byteorder='big')
    if (progress is not None):
        progress(0, count)
    rows = int.from_bytes(f.read(4), byteorder='big')
    cols = int.from_bytes(f.read(4), byteorder='big')
    index = 0
    data = []
    bytes = f.read(rows * cols)
    while (bytes):
        img = array('B', bytes)
        data.append(img)
        index += 1
        bytes = f.read(rows * cols)
        if (progress is not None):
            progress(index, count)
        if (max != 0 and index >= max):
            break

    if count != index:
        print("Warning: file broken... counts doesn't match (given count: %s in data: %s)" % (count, index))

    return np.array(data)

def calc_distance(array1, array2):
    return np.subtract(array1, array2).power(2).sum()


def show_progress(current, max, text):
    if (current == max):
        print("{}\r%s finished!" % (text))
        return

    oldpercent = round(float(current-1)/max, 3)
    percent = round(float(current)/max, 3)
    if (1 > percent > oldpercent or current == 0):
        print("{}\r%s %0.1f %%" % (text, percent * 100), end="")

    return

def progress(text):
    return lambda current, max: show_progress(current, max, text)

def k_nn(k, train_labels, train_images, image):

    distances = []
    index = 0
    while (index < len(train_images)):
        dst = calc_distance(image, train_images[index])
        lbl = train_labels[index]
        distances.append((dst, lbl))
        index += 1

    k_shortest = sorted(distances, key=lambda x: x[0])[0:k]
    occurences = [0] * 10
    for distance in k_shortest:
        occurences[distance[1]] += 1

    label = -1
    best = 0
    lbl = 0
    while (lbl < len(occurences)):
        if (occurences[lbl] > best):
            label = lbl
        lbl += 1

    return label

def k_means(k, images, iterations):
    means = []
    pixel_count = len(images[0])
    for i in range(k):
        mean = array('B', [0] * pixel_count)
        for x in range(pixel_count):
            mean[x] = random.randint(0, 255)
        means.append(mean)

    groups = []
    for i in range(iterations):
        groups = []
        for a in range(k):
            groups.append([])
        index = 0
        # assign image to mean
        for image in images:
            distances = []
            for mean in means:
                distances.append(calc_distance(mean, image))
            group = index_min(distances)
            groups[group].append(index)
            index += 1

        # recalculate means
        for i in range(len(means)):
            if (len(groups[i]) != 0):
                means[i] = calc_mean(groups[i], images)

    return groups




def calc_mean(group, images):
    count = len(group)
    mean = [0] * len(images[0])
    for index in group:
        image = images[index]
        for x in range(len(image)):
            mean[x] += image[x]/count

    for x in range(len(mean)):
        mean[x] = mean[x] / count

    return mean


def index_min(list):
    index = 0
    min_index = 0
    min = float("inf")
    for item in list:
        if (item < min):
            min = item
            min_index = index
        index += 1
    return min_index

def draw_array(pixels, width, height):
    count = 0
    pixel = pixels[0]
    print("listentyp: %s elementtyp: %s" % (type(pixels), type(pixel)))
    if isinstance(pixel, (tuple, list, np.ndarray, array)):
        image = Image.new('RGB', (width, height), (255,255,255))
    else:
        image = Image.new('L', (width, height), 255)

    while count < width * height:
        pixel = pixels[count]
        y = int(count / width)
        x = int(count % width)
        if isinstance(pixel, (tuple, list, np.ndarray, array)):
            image.putpixel((x, y), (int(pixel[0]), int(pixel[1]), int(pixel[2])))
        else:
            image.putpixel((x, y), int(pixel))
        count += 1

    return image


def draw_matrix(matrix):
    width = len(matrix[0])
    height = len(matrix)
    pixel = matrix[0,0]

    if isinstance(pixel, (tuple, list, np.ndarray, array)):
        image = Image.new('RGB', (width, height), (255,255,255))
    else :
        image = Image.new('L', (width, height), 255)

    x = 0; y = 0
    while y < height:
        while x < width:
            pixel = matrix[y, x]
            if isinstance(pixel, (tuple, list, np.ndarray, array)):
                image.putpixel((x, y), (int(pixel[0]), int(pixel[1]), int(pixel[2])))
            else:
                image.putpixel((x, y), int(pixel))
            x += 1
        y += 1
        x = 0

    return image

def show_images(images, rows = 2, title = "Bilder"):
    root = Tk()
    root.title(title)

    count = len(images)
    width = math.ceil(count / rows)
    tkimages = []
    for image in images:
        tkimages.append(ImageTk.PhotoImage(image))

    i = 0
    for image in images:
        x = int(i / width)
        y = int(i % width)
        Label(root, image=tkimages[i], text=i).grid(row=x, column=y)
        i += 1

    root.mainloop()
    return root

def show_image(image, global_root = None, title = "Bild"):
    root = Tk()


    root.title(title)
    root.geometry("%sx%s" % (image.size[0], image.size[1]))
    tkimage = ImageTk.PhotoImage(image)
    lblimage = Label(root, image=tkimage)
    lblimage.pack()

    root.mainloop()

    return root

def stats(calc_labels, test_labels):
    index = 0
    right = 0
    fail = 0
    while(index < len(test_labels)):
        if (test_labels[index] == calc_labels[index]):
            right += 1
        else:
            fail += 1
        index += 1

    return (right, fail)

max = 0
test_labels = load_labels(r'/data/t10k-labels.idx1-ubyte', progress("loading test data labels..."), max)
test_images = load_images(r'/data/t10k-images.idx3-ubyte', progress("loading test data images..."), max)
train_labels = load_labels(r'/data/train-labels.idx1-ubyte', progress("loading training data labels..."), max)
train_images = load_images(r'/data/train-images.idx3-ubyte', progress("loading training data images..."), max)

print("\n\n------------\n\nRANDOM VALUES:\n")

rand_idx = random.sample(range(0, len(test_images)), 10)
images = []
for index in rand_idx:
    img = draw_array(test_images[index], 28, 28)
    images.append(img)

show_images(images)

#root = show_images(images, 2, False)
    #print("index: %s, label: %s, data: %s" % (index, test_labels[index], test_images[index]))

index = len(test_images) - 1
print("\n\n------------\n\nAUFGABE 1:\n")
print("index: %s, label: %s, data: %s" % (index, test_labels[index], test_images[index]))

print("\n\n------------\n\nAUFGABE 2: k-NN\n")


count = len(test_images)
print("k = 1")
k1_labels = []
i = 0
for test_image in test_images:
    show_progress(i, count, "calculating nearest neighbors (k = 1)...")
    k1_labels.append(k_nn(1, train_labels, train_images, test_image))
    i += 1

print("\n\nstatistic (k = 1):")
stat = stats(k1_labels, test_labels)
print("right : %0.2f%% (%s)    -    wrong: %0.2f%% (%s)" % ((float(stat[0]) / count) * 100, stat[0], (float(stat[1]) / count) * 100, stat[1]))


print("\n---\nk = 3")
k3_labels = []
i = 0
for test_image in test_images:
    show_progress(i, count, "calculating nearest neighbors (k = 3)...")
    k3_labels.append(k_nn(3, train_labels, train_images, test_image))
    i += 1

print("\n\nstats (k = 3):")
stat = stats(k3_labels, test_labels)
print("right : %0.2f%% (%s)    -    wrong: %0.2f%% (%s)" % ((float(stat[0]) / count) * 100, stat[0], (float(stat[1]) / count) * 100, stat[1]))


print("\n---\nk = 11")
k11_labels = []
i = 0
for test_image in test_images:
    show_progress(i, count, "calculating nearest neighbors (k = 11)...")
    k11_labels.append(k_nn(11, train_labels, train_images, test_image))
    i += 1

print("\n\nstats (k = 11):")
stat = stats(k11_labels, test_labels)
print("right : %0.2f%% (%s)    -    wrong: %0.2f%% (%s)" % ((float(stat[0]) / count) * 100, stat[0], (float(stat[1]) / count) * 100, stat[1]))


print("\n\n------------\n\nAUFGABE 3: k-means\n")
print("\nk=9\n")
k9_groups = k_means(9, test_images, 2)
index = 0
for group in k9_groups:
    print("Cluster %s:" % (index))
    index += 1
    for i in range(10):
        if (len(group) > i):
            print("index: %s data: %s " % (group[i], test_images[group[i]]))
    print('\n\n')


print("---")
print("\nk=10\n")
k10_groups = k_means(10, test_images, 10)
index = 0
for group in k10_groups:
    print("Cluster %s:" % (index))
    index += 1
    for i in range(10):
        if (len(group) > i):
            print("index: %s data: %s " % (group[i], test_images[group[i]]))
    print('\n\n')


print("---")
print("\nk=20\n")
k20_groups = k_means(20, test_images, 10)
index = 0
for group in k20_groups:
    print("Cluster %s:" % (index))
    index += 1
    if (len(group) == 0):
        print("<leer>")
    for i in range(10):
        if (len(group) > i):
            print("index: %s data: %s " % (group[i], test_images[group[i]]))
    print('\n\n')


