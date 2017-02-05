# -*- coding: utf-8 -*-
"""
Created on Sun Jan 29 10:31:47 2017

@author: KimuKook
"""

import numpy as np

def theoritical_sigma(corr,s,sigma02):
    sigma = np.zeros([len(s)])
    for ii in range(len(s)):
        sigma[ii] = 1
        for jj in range(1,s[ii]):
            sigma[ii] = sigma[ii] + 2*(1-jj/s[ii])*corr[jj-1]
        sigma[ii] = sigma02 * sigma[ii] / s[ii]
    return sigma