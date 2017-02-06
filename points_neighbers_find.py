

import numpy as np

np.set_printoptions(linewidth=200, precision=5, suppress=True)
import pandas as pd;

pd.options.display.max_rows = 20;
pd.options.display.expand_frame_repr = False
import pylab as plt
import numpy.matlib

import Utils as utl


def points_neighbers_find(x,xE,xU):

	[delta_general, index,x1] = utl.mindis(x, np.concatenate((xE,xU ), axis=0) )

	


	return x, xE, xU, newadd, sucess	




#function [x,xE,xU,newadd,sucess]=points_neighbers_find(x,xE,xU)
#% Add the new point to the set
#% xE: evaluation points.
#% xU: unevaluated points.
#%
#
#global n Ain bin
#
#%keyboard
#% Find closest point to x
#[del_general,index,x1]=mindis(x,[xE xU]);
#
#
#% Caclulate the active constraints at x and x1
#%keyboard
#active_cons=1:2*n; b=bin-Ain*x; active_cons=active_cons(b<1e-3); 
#active_cons1=1:2*n; b=bin-Ain*x1; active_cons1=active_cons1(b<1e-3); 
#
#% Check the closest point is acceptable 
#   if (isempty(active_cons) || min(ismember(active_cons,active_cons1))==1)
#    % Acceptable case
#           newadd=1;
#           sucess=1;
#           if mindis(x,xU)==0
#               newadd=0;
#           end
#    % New point is close to xU, and x1 is closer to xE. 
#       %  if index>size(xE)
#       %    newadd=0; x=x1;
#       %    xU(:,index-size(xE))=[];
#       %  end
#   else
#    % unacceptable case
#    sucess=0;
#    newadd=0; xU=[xU x]; 
#   end
#
#end
#