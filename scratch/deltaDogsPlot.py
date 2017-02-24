from pylab import *
from scipy.interpolate import interp1d,UnivariateSpline

def f(x):
  return  -x*sin((1.5*2*3.14*x))+1

def e(x,(x1,x2)):
  return -(x-x1)*(x-x2)

def e2d(x,y,xi,yi):
  rx,ry=x-xi,y-yi; 
  r = sqrt(rx**2+ry**2)
  return r.prod()


if __name__ == "__main__":
  ion();

  x = linspace(0,1,1000);
  yf = f(x);

  y0 = yf.min()-1e-2

  xi = array([0,.72,1]);
    

  for i in xrange(0,1):
    yi = f(xi)


    #p = interp1d(xi, yi,kind='quadratic');
    p = UnivariateSpline(xi,yi,k=2,s=0);
    yp = p(x);


    ye = array([])
    for j in xrange(1,len(xi)):
      test = np.where(np.logical_and(x>=xi[j-1], x<xi[j]))
      yetmp = e(x[ test ] , ( xi[j] , xi[j-1] ) )
      ye = hstack((ye,yetmp))
    ye = hstack((ye,0))

    ys = (yp-y0)/ye;



    subplot(211)
    cla() 
    plot(x,yf,'k');
    plot(xi,yi,'ok')
    plot(x,yp,'b');
    plot(x,ye,'r');
    subplot(212)
    cla() 
    semilogy(x,ys);

    draw()

    #print "data points\n",vstack((xi,yi)).T
    xmin,ymin = x[argmin(ys[1:-1])+1],amin(ys[1:-1])
    subplot(211)
    plot(xmin,f(xmin),'or')
    xi = sort(append(xi,xmin))

    print "minimum of search function",(xmin,ymin)
    raw_input("press a key to exit");

  data = vstack((x,yf,yp,ye,ys))
  savetxt("deltaDogsPlot.dat",data.T)
  data = vstack((xi,f(xi)))
  savetxt("deltaDogsPlotNodes.dat",data.T)
  #show()
  raw_input("press a key to exit");

