import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as pl
# import matplotlib.pyplot as plt
import pylab as plt




data1FilePath = "data1.txt"
data2FilePath = "data2.txt"


def readInputFile(filePath):
    #    retVal = []
    #    with open(filePath, 'rb') as csvfile:
    #        filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    #        for row in filereader:
    #            retVal.append([int(row[0]), int(row[1]), int(row[2])])

    retVal = []
    with open(filePath) as file:
        line = file.readline()
        arr = [float(a) for a in line.split(',')]
        #        retVal.append(file.readline())
        retVal.append(arr)
    return retVal[0]


def transient_removal(x=[]):
    N = len(x)
    k = np.int_([N / 2])
    y = np.zeros((k, 1))
    for kk in np.arange(k):
        y[kk] = np.var(x[kk + 1:]) * 1.0 / (N - kk - 1.0)
    y = np.array(-y)
    ind = y.argmax(0)
    print('index of transient point in the signal:')
    print(ind)
    return ind


x = readInputFile(data1FilePath)
x = x[:10000]
index = transient_removal(x)


## sampled time intervals 1
t = np.arange(0., len(x))
# red dashes transient detector, green curve simulation results of KSE
plt.plot(t, x)
plt.plot([index,index], [np.min(x)/2.0, np.max(x)], '--r')
plt.show()

