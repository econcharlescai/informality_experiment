#Steps

#1. Define profit as a function of theta, w and p
#2. Equate this to zero to find profit cutoff
#3. Find the value function of a given signal
#4. Find labor demand

import numpy as np
from scipy.optimize import brentq
from scipy.optimize import fminbound
import scipy.integrate as integrate
from scipy.stats import lognorm
from math import log


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
sig = 0.245
entrycf = 15
entryci = 10
numin = 7.7
numax = 100
xi = 3.9


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

    return -minimand - ff

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

cutoffthei = brentq(profiti, themin, themax)
cutoffthef = brentq(profitf, themin, themax)

def vali(the):
    return max(0, profiti(the)/kapi)

def valf(the):
    return max(0, profitf(the)/kapf)

def valsigi(nu):
    result = integrate.quad(lambda x: vali(x)*lognorm.pdf(x-log(nu), sig), 1,10)
    return result[0]

def valsigf(nu):
    result = integrate.quad(lambda x: valf(x)*lognorm.pdf(x-log(nu), sig), 1,10)
    return result[0]

entrynui = brentq(lambda x: valsigi(x)-entryci, numin, numax)
print(entrynui)

# entrynuf = brentq(lambda x: valsigf(x)-entrycf, numin, numax)
# print(entrynuf)

cutoffnu = brentq(lambda x: (valsigf(x)-entrycf) - (valsigi(x)-entryci), numin, numax)
print(cutoffnu)

def df(nu):

    if nu < cutoffnu:
        return 0
    else:
        result = integrate.quad(lambda x: (max(laborf(x)-lbar,0))*lognorm.pdf(x-log(nu),sig), 1, 10, epsabs = 1e-5)
        return result[0]


def di(nu):

    if nu < entrynui:
        return 0
    elif nu < cutoffnu:
        result = integrate.quad(lambda x: labori(x)*lognorm.pdf(x-log(nu),sig), 1, 10, epsabs = 1e-5)
        return result[0]
    else:
        result = integrate.quad(lambda x: min(lbar,laborf(x))*lognorm.pdf(x-log(nu),sig), 1, 10, epsabs = 1e-5)
        return result[0]


demandi = integrate.quad(lambda x: di(x)*xi*(x**(-xi-1)) * numin**(xi), numin, np.inf, epsabs = 1e-5)
demandf = integrate.quad(lambda x: df(x)*xi*(x**(-xi-1)) * numin**(xi), numin, np.inf, epsabs = 1e-5)
print(demandi)
print(demandf)

# print(valsigi(7.7))
# print(valsigi(100))
