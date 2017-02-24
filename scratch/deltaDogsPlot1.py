from pylab import *
from scipy.interpolate import interp1d,UnivariateSpline

def f(x):
  return  -x*sin((1.5*2*3.14*x))+1

def e(x,(x1,x2)):
  return -(x-x1)*(x-x2)



if __name__ == "__main__":
  ion();
  y0 = 0

  x = linspace(0,1,1000);
  yf = f(x);


  xi = array([0,.72,1]);
    

  for i in xrange(0,100):
    yi = f(xi)


    p = interp1d(xi, yi,kind='quadratic');
    #spl = UnivariateSpline(xi,yi,k=2);
    yp = p(x);


    ye = array([])
    for j in xrange(1,len(xi)):
      test = np.where(np.logical_and(x>=xi[j-1], x<xi[j]))
      yetmp = e(x[ test ] , ( xi[j] , xi[j-1] ) )
      ye = hstack((ye,yetmp))
    ye = hstack((ye,0))

    yu = (yp-y0)/ye;



    subplot(211)
    cla() 
    plot(x,yf,'k');
    plot(xi,yi,'ok')
    plot(x,yp,'b');
    plot(x,ye,'r');
    subplot(212)
    cla() 
    semilogy(x,yu);

    draw()

    #print "data points\n",vstack((xi,yi)).T
    xmin,ymin = x[argmin(yu[1:-1])+1],amin(yu[1:-1])
    subplot(211)
    plot(xmin,f(xmin),'or')
    xi = sort(append(xi,xmin))

    print "minimum of search function",(xmin,ymin)
    raw_input("press a key to exit");

  #data = vstack((x,yf,yp,ye,yu))
  #savetxt("deltaDogsPlot.dat",data.T)
  #show()
  raw_input("press a key to exit");

