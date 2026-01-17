import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng()
dt_max = .0001 
sqrtdt_max = np.sqrt(dt_max)
T = 1.  # Total time.
n = int(T / dt_max)  # Number of time steps.
w_max = rng.normal(0.0, sqrtdt_max, n) 
     
a = 0.3
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

def Schemer(dt, X, f, g, dgdx, wmax, Scheme, dtmax=dt_max):
    factor = int(dt/dt_max)
    w = wmax[::factor]
    dw = w[1:] - w[:-1]
    t = np.arange(0, 1, dt)[1:]
    xlist = []

    for i in range(len(t)):
        xlist.append(Scheme(X, f, g, dgdx, t[i], dt, dw[i]))

    xlist = np.array(xlist)
    plt.plot(t, xlist[:, 1])
    return t, xlist



for i in range(1, 20):
    Schemer(i*dt_max*5, X0, itof, itog, itodgdx, w_max, MilsteinStep)





#Euler-Maruyama
#for i in range(n - 1):
#    w[i + 1] = w[i]+ sqrtdt * np.random.randn()
#    x[i + 1] = x[i] + y[i]*dt
#    y[i + 1] = y[i] + (-b*y[i] - np.sin(x[i]) + c*np.sin(2*x[i]))*dt + (-a*b*y[i] - a*np.sin(x[i]))*(w[i+1] - w[i])
    
#Milstein
#for i in range(n - 1):
#    w[i + 1] = w[i]+ sqrtdt * np.random.randn()
#    x[i + 1] = x[i] + y[i]*dt
#    y[i + 1] = y[i] + (-b*y[i] - np.sin(x[i]) + c*np.sin(2*x[i]))*dt + (-a*b*y[i] - a*np.sin(x[i]))*(w[i+1] - w[i]) - (-a*b*y[i] - a*np.sin(x[i]))*a*b*((w[i + 1] - w[i])**2 - dt) 
#    
#
#plt.plot(t, w_max)
#plt.plot(t, x)
#plt.plot(t, y)

#fig, ax = plt.subplots(1, 1, figsize=(8, 4))
#ax.plot(tc, xc, lw=2)
#ax.plot(tc,gc,'ro',markersize=1)

#
#ntrials = 10000
#X = np.zeros(ntrials)
#     
#bins = np.linspace(-2., 14., 100)
#fig, ax = plt.subplots(1, 1, figsize=(8, 4))
#for i in range(n):
#    # We update the process independently for
#    # all trials
#    X +=  b*b*dt*X*0.5+sqrtdt * np.random.randn(ntrials)
#    # We display the histogram for a few points in
#    # time
#    if i in (5, 50, 900):
#        hist, _ = np.histogram(X, bins=bins)
#        ax.plot((bins[1:] + bins[:-1]) / 2, hist,
#                {5: '-', 50: '.', 900: '-.', }[i],
#                label=f"t={i * dt:.2f}")

plt.show()
