
import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt
import numpy.matlib


# import Utils as utl

def ismember(A ,B):
	return [np.sum(a == B) for a in A]

def points_neighbers_find(x ,xE ,xU ,Bin ,Ain):
	[delta_general, index ,x1] = mindis(x, np.concatenate((xE ,xU ), axis=1) )

	active_cons = []
	b = Bin - np.dot(Ain ,x)
	for i in range(len(b)):
		if b[i][0] < 1e-3:
			active_cons.append( i +1)
	active_cons = np.array(active_cons)

	active_cons1 = []
	b = Bin - np.dot(Ain ,x1)
	for i in range(len(b)):
		if b[i][0] < 1e-3:
			active_cons1.append( i +1)
	active_cons1 = np.array(active_cons1)

	if len(active_cons) == 0 or min(ismember(active_cons ,active_cons1)) == 1:
		newadd = 1
		success = 1
		if mindis(x ,xU) == 0:
			newadd = 0
	else:
		success = 0
		newadd = 0
		xU = np.concatenate((xU ,x) ,axis=0)
	return x, xE, xU, newadd, success

def mindis(x, xi):
	# function [y,x1,index] = mindistance(x,xi)
	# % calculates the minimum distance from all the existing points
	# % xi all the previous points
	# % x the new point
	y = float('inf')
	N = xi.shape[1]
	for i in range(N):
		y1 = np.linalg.norm(x[:, 0] - xi[:, i])
		if y1 < y:
			y = np.copy(y1)
			x1 = np.copy(xi[:, i])
			index = np.copy(i)
	return y, index, x1


# %%
Ain = np.array([[1 ,2 ,1 ,0] ,[2 ,1 ,2 ,1] ,[1 ,0 ,2 ,1] ,[1 ,2 ,1 ,1]])
Bin = np.array([[1] ,[2] ,[1] ,[2]])
x = np.array([[1] ,[2] ,[1] ,[4]])
xE = np.array([[2] ,[4] ,[1] ,[2]])
xU = np.array([[1] ,[0] ,[1] ,[2]])
n = 2
x ,xE ,xU ,newadd ,success = points_neighbers_find(x ,xE ,xU ,Bin ,Ain)
print(success)