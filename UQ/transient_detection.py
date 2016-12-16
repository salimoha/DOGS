from __future__ import division
import csv
import numpy as np

data1FilePath = "data1.txt"
data2FilePath = "data2.txt"

def readInputFile(filePath):
#    retVal = []
#    with open(filePath, 'rb') as csvfile:
#        filereader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#        for row in filereader:
#            retVal.append([int(row[0]), int(row[1]), int(row[2])])

    retVal=[]
    with open(filePath) as file:
         line=file.readline()
         arr=[float(a) for a in line.split(',')]
 #        retVal.append(file.readline())
         retVal.append(arr)
    return retVal[0]

def transient_detection(x=[]):
    N = len(x)
    k = np.int_([N/2])
    y = np.zeros((k, 1))
    for kk in np.arange(k):
        y[kk] = np.var(x[kk+1:])*1.0/(N-kk-1.0)
    y = np.array(-y)
    ind = y.argmax(0)
    print('index of transient point in the signal:')
    print(ind)
    return ind

x = readInputFile(data1FilePath)
x = x[:10000]
index = transient_removal(x)