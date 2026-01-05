import numpy as np
import matplotlib.pyplot as plt


dt = .001 
T = 1.  # Total time.
n = int(T / dt)  # Number of time steps.
t = np.linspace(0., T, n)  # Vector of times.
sqrtdt = np.sqrt(dt)
     
w = np.zeros(n)

x = np.zeros(n)
y = np.zeros(n)
y[0] = .1

a = 1.3
b = 0.6
c = 0.3

for i in range(n - 1):
    w[i + 1] = w[i]+ sqrtdt * np.random.randn()
    x[i + 1] = x[i] + y[i]*dt
    y[i + 1] = y[i] + (-b*y[i] - np.sin(x[i]) + c*np.sin(2*x[i]))*dt + (-a*b*y[i] - a*np.sin(x[i]))*(w[i+1] - w[i])
    
    
plt.plot(t, w)
#plt.plot(t, x)
plt.plot(t, y)

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
