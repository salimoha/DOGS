
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from UQ import stationary_statistical_learning_reduced

def lorenz(x, y, z, s=10, r=28, b=2.667):
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot

def objfun(s,r,b,stepCnt = 1000, dt = 0.01):
    # Need one more for the initial values
    xs = np.empty((stepCnt + 1,))
    ys = np.empty((stepCnt + 1,))
    zs = np.empty((stepCnt + 1,))
    # Setting initial values
    xs[0], ys[0], zs[0] = (0., 1., 1.05)
    # xs[0], ys[0], zs[0] = np.random.rand(3)
    # Stepping through "time".
    for i in range(stepCnt):
        # Derivatives of the X, Y, Z state
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], s, r, b)
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)
    # mu =np.mean(zs)
    # mu = np.vstack((mu,np.mean(zs)))
    return np.mean(zs)#, np.mean(zs**2), np.mean(xs)

s,r,b =(10.,28.,2.667)
T = int(1e4)
h=0.01
iter_max = 20
# m1,m2,m3 = objfun(s,r,b,T,h)
#
# for k in range(iter_max):
#     f = objfun(s,r,b,T,h)
#     mu = np.vstack((mu,f))
# print(mu.mean())
# s2 = stationary_statistical_learning_reduced(mu,4)
# print(s2[0])


########### beta

mb =objfun(s,r,b,T,h)
for bb in np.arange(0.3,12,0.1):
    mb = np.vstack((mb,objfun(s, r, bb, T, h)))

fig = plt.figure()
plt.plot(np.arange(0.3,12,0.1),mb[1:])
plt.plot(np.array([b]),mb[0],'*')
#plt.show()


########### sigma
ms =objfun(s,r,b,T,h)
for ss in np.arange(1,14,0.1):
    ms = np.vstack((ms,objfun(ss, r, b, T, h)))

fig = plt.figure()
plt.plot(np.arange(1,14,0.1),ms[1:])
plt.plot(np.array([s]),ms[0],'*')
# plt.show()

########### r
mr =objfun(s,r,b,T,h)
for rr in np.arange(1,40,0.1):
    mr = np.vstack((mr,objfun(s, rr, b, T, h)))

fig = plt.figure()
plt.plot(np.arange(1,40,0.1),mr[1:])
plt.plot(np.array([r]),mr[0],'*')
plt.show()

# a = objfun(10,28,2.667,T,h); print(a)

#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
#
# ax.plot(xs, ys, zs, lw=0.5)
# ax.set_xlabel("X Axis")
# ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
# ax.set_title("Lorenz Attractor")
#
# plt.show()