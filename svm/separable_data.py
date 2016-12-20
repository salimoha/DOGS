import numpy as np
import matplotlib.pyplot as plt
import time 
#pylab inline

numpy.random.seed(2016)

X1 = np.random.randn(100)+1.5
Y1 = -2*(X1)+8+2*np.random.randn(100)
X2 = np.random.randn(100)-1.5
Y2 = -2*(X2)-4 + 2*np.random.randn(100)

plt.plot(X1,Y1,'s',X2,Y2,'o')
plt.ylim(-8,8)
plt.xlim(-8,8)
