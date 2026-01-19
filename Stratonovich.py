import numpy as np
import matplotlib.pyplot as plt

def f_ito(y, b, c):
    x, v = y
    return np.array([v, -b*v - np.sin(x) + c*np.sin(2*x)])

def g(y, a, b):
    x, v = y
    return np.array([0.0,-a*(b*v + np.sin(x))])

def drift_correction(y, a, b):
    x, v = y
    return np.array([0.0,0.5*(a**2)*b*(b*v + np.sin(x))])

def f_strat(y, a, b, c):
    return f_ito(y, b, c) + drift_correction(y, a, b)

#for  testing reasons
def Maruyama(x0, v0, T, N, a, b, c, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / N
    BeginConditions = np.array([x0, v0])
    Result = np.empty((N+1, 2))
    Result[0] = BeginConditions
    dW = rng.normal(0,np.sqrt(dt),N)
    for i in range(N):
        Result[i+1] = Result[i] + f_ito(Result[i], b, c)*dt + g(Result[i], a, b)*dW[i]
    return Result

def Stratonovich(x0, v0, T, N, a, b, c, seed=np.random.randint(0,2**30,1)):
    rng = np.random.default_rng(seed)
    dt = T / N
    BeginConditions = np.array([x0, v0])
    Result = np.empty((N+1, 2))
    Result[0] = BeginConditions
    dW = rng.normal(0,np.sqrt(dt),N)
    for i in range(N):
        Result[i+1] = Result[i] + f_strat(Result[i], a, b, c)*dt + g(Result[i], a, b)*dW[i]
    return Result



if __name__ == "__main__":
    a = 0.3
    b = 0.6
    c = 0.3

    x0, v0 = 0.5*np.pi, 0.0
    T = 2
    N = 50000*50
    space = np.linspace(0,T,num=N+1)

    



    ito_path = Maruyama(x0, v0, T, N, a, b, c, seed=30)
    strat_path = Stratonovich(x0, v0, T, N, a, b, c, seed=30)
    print(space.shape)
    plt.plot(space,ito_path[:,0],label="Ito")
    plt.plot(space,strat_path[:,0],label="strat")
    plt.title(r"Comparing IT$\^{O}$ and Stratonovich")
    plt.grid(True, which="both", alpha=0.3)
    plt.ylabel("angle")
    plt.xlabel("time")
    plt.legend()
    # plt.plot(space,strat_path[:,0])
    plt.savefig('Strato.png')

