#Steps

#1. Define profit as a function of theta, w and p
#2. Equate this to zero to find profit cutoff

import numpy as np
from scipy.optimize import brentq
from scipy.optimize import fminbound
from scipy.integrate import integrate


alp = 0.65
p = 1
w = 1
bi = 5.01
lmin = 0.01
lmax = 100
fi = 0.248
themin = 1
themax = 10
bf = 4
lbar = 1
tauw = 0.375
tauy = 0.293
ff = 0.5
kapf = 0.129
kapi = 0.349

def labori(the):

    def foc(x):
        return p*the*alp*(x**(alp-1)) - w - 2*w*(bi**(-1))*x

    return brentq(foc, lmin, lmax)

def profiti(the):

    l = labori(the)

    return p*the*l**alp - (1 + (l/bi))*l*w - fi

def profitf(the):

    def costf(x):

        if x < lbar:
            return w*(1 + (x/bf))*x
        else:
            return w*(1 + (lbar/bf))*lbar + (1 + tauw)*w*(x-lbar)

    def payofff(x):

        return -(1-tauy)*the*x**(alp) + costf(x)

    minimiser, minimand, ierr, numfunc =  fminbound(payofff, lmin, lmax, full_output=1)

    return minimand - ff

def laborf(the):

    def costf(x):

        if x < lbar:
            return w*(1 + (x/bf))*x
        else:
            return w*(1 + (lbar/bf))*lbar + (1 + tauw)*w*(x-lbar)

    def payofff(x):

        return -(1-tauy)*the*x**(alp) + costf(x)

    minimiser, minimand, ierr, numfunc =  fminbound(payofff, lmin, lmax, full_output=1)

    return minimiser

cutoffi = brentq(profiti, themin, themax)
cutofff = brentq(profitf, themin, themax)

def vali(the):
     return max(0, profiti(the)/kapi)

def valf(the):
    return max(0, profitf(the)/kapf)
