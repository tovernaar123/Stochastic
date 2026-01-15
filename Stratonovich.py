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
    return np.array([0.0,0.5*(a**2)*b*(b*v + np.sin(x))], dtype=float)

def f_strat(y, a, b, c):
    return f_ito(y, b, c) - drift_correction(y, a, b)

#for  testing reasons
# def Maruyama(x0, v0, T, N, a, b, c, seed=0):
#     rng = np.random.default_rng(seed)
#     dt = T / N
#     dW = rng.normal(0,np.sqrt(dt),N)
#     y = np.array([x0, v0], dtype=float)
#     path = np.empty((N+1, 2), dtype=float)
#     path[0] = y

#     for n in range(N):
#         y = y + f_ito(y, b, c) * dt + g(y, b, c) * dW[n]
#         path[n+1] = y
#     return path

def Stratonovich(x0, v0, T, N, a, b, c, seed=np.random.random_integers(0,2**30,1)):
    rng = np.random.default_rng(seed)
    dt = T / N
    BeginConditions = np.array([x0, v0])
    Result = np.empty((N+1, 2))
    Result[0] = BeginConditions
    dW = rng.normal(0,np.sqrt(dt),N)
    for i in range(N):
        y_tilde = Result[i] + f_strat(Result[i], a, b, c)*dt + g(Result[i], a, b)*dW[i]
        Result[i+1] = Result[i] + 1/2 *(f_strat(Result[i], a, b, c) +f_strat(y_tilde, a, b, c) )*dt + 1/2*(g(Result[i], a, b)+ g(y_tilde, a, b))*dW[i]
    return Result



if __name__ == "__main__":
    a = 0.2
    b = 0.6
    c = 0.2
    x0, v0 = 0.1, 0.0
    T = 50.0
    N = 50000
    space = np.linspace(0,T,num=N+1)

    # ito_path = Maruyama(x0, v0, T, N, a, b, c, seed=30)
    strat_path = Stratonovich(x0, v0, T, N, a, b, c)
    print(space.shape)
    # plt.plot(space,ito_path[:,1])
    plt.plot(space,strat_path[:,1])
    plt.show()

