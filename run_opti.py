import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt
# import Alpha DOGS
import Utils as dg

def opt_init():

    # read initialization points
    # config file
    # Muhan .....??? TODO
    n  = 3
    lb = 0
    ub = 1

    xU = dg.bounds(lb,ub,n) # support points
    xE = np.random.rand(n,n+1) # TODO: normalize the data 0,1

    xT = pd.DataFrame(xE)
    xT.to_csv("pts_to_eval.csv",index=False)
    xT.to_csv("Xall.csv", index=False)
    k=0
    np.savetxt('iter_his.txt',k)


def solver(x):
    trans =1
    ind_prev = 0
    while trans ==1:
        y = fun(x[:T])
        ind = transient_detector(y)

# opt_init()
# for k in range(iter_max):
while stop == 0

    Alpha_DOGS()

    #read the file from the pts_to_eval
    xEval = pd.read_csv("pts_to_eval.csv")

    # function evaluation
    for i in range(xEval.shape[1]):

        yEval[i], sigma[i],  = solver(xEval[:,i])

    yE = pd.DataFrame(npyEval)
    yE.to_csv("surr_yi_new")
    yE.to_csv("Yall.csv", mode='w+') # TODO: write the yi to the end of function

    stop = np.load("optStop.txt")