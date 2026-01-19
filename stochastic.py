import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
T = 5.0
n = 2**14 + 1
dt_max = T/(n-1)
sqrtdt_max = np.sqrt(dt_max)
dwmax = rng.normal(0.0, sqrtdt_max,  n-1)
w_max = np.zeros(n)
w_max[1:] = np.cumsum(dwmax)
     
a = 0.8
b = 0.6
c = 0.3

X0 = np.array([0.2, 0.1])

def itof(X, t):
    x = X[0]
    y = X[1]
    return np.array([y, -b*y - np.sin(x) + c*np.sin(2*x)])

def itog(X, t):
    x = X[0]
    y = X[1]
    return np.array([0, -a*b*y - a*np.sin(x)])

def itodgdx(X, t):
    x = X[0]
    y = X[1]
    return np.array([[0, 0], [-a*np.cos(x), -a*b]])

def EulerMaruyamaStep(X, f, g, dgdx, t, dt, dWt):
    return X + f(X, t)*dt + g(X, t)*dWt

def MilsteinStep(X, f, g, dgdx, t, dt, dWt):
    gvar = g(X, t)
    dval = np.matmul(dgdx(X, t), gvar)
    return X + f(X, t)*dt + gvar*dWt + 0.5*dval*(dWt*dWt - dt)

def Schemer(factor, X, f, g, dgdx, wmax, Scheme):
    X = np.array(X, dtype=float)
    w = wmax[::factor]
    dw = np.diff(w)
    dt = dt_max * factor
    t = dt * np.arange(1, len(dw) + 1)
    xlist = np.empty((len(dw), 2), dtype=float)
    for i in range(len(dw)):
        X = Scheme(X, f, g, dgdx, t[i], dt, dw[i])
        xlist[i] = X
    return t, xlist

def StrongError(factorlist, X, f, g, dgdx, Scheme, trials):
    dwmax = rng.normal(0.0, sqrtdt_max, (trials, n-1))
    wmax = np.zeros((trials, n))
    wmax[:, 1:] = np.cumsum(dwmax, axis=1)
    errorlist = []
    for factor in factorlist:
        errs = []
        for i in range(trials):
            _, fine = Schemer(1, X, f, g, dgdx, wmax[i], Scheme)
            _, coarse = Schemer(factor, X, f, g, dgdx, wmax[i], Scheme)
            ref = fine[factor-1::factor, 1]
            errs.append(abs(ref[-1] - coarse[-1, 1]))
        errorlist.append(np.mean(errs))
    return np.array(errorlist)

def WeakError(factorlist, X, f, g, dgdx, Scheme, trials):
    dwmax = rng.normal(0.0, sqrtdt_max, (trials, n-1))
    wmax = np.zeros((trials, n))
    wmax[:, 1:] = np.cumsum(dwmax, axis=1)
    fine_vals = []
    for i in range(trials):
        _, fine = Schemer(1, X, f, g, dgdx, wmax[i], Scheme)
        fine_vals.append(fine[-1, 1])
    fine_mean = np.mean(fine_vals)
    errorlist = []
    for factor in factorlist:
        coarse_vals = []
        for i in range(trials):
            _, coarse = Schemer(factor, X, f, g, dgdx, wmax[i], Scheme)
            coarse_vals.append(coarse[-1, 1])
        errorlist.append(abs(fine_mean - np.mean(coarse_vals)))
    return np.array(errorlist)

factors = np.array([2,4,8,16,32,64,128])
trials = 100
plt.figure()
plt.xscale('log')
plt.yscale('log')
plt.plot(dt_max*factors, StrongError(factors, X0, itof, itog, itodgdx, EulerMaruyamaStep, trials))
plt.plot(dt_max*factors, WeakError(factors, X0, itof, itog, itodgdx, EulerMaruyamaStep, trials))
plt.show()
