import time
import numpy as np
import matplotlib.pyplot as plt

a = 0.3
b = 0.6
c = 0.3

def itofv(x, v, sx):
    return -b*v - sx + c*np.sin(2*x)

def stratfv(x, v, sx):
    return itofv(x, v, sx) + 0.5*a*a*b*(b*v + sx)

def gv(x, v, sx):
    return -a*(b*v + sx)

def EulerMaruyamaStep(x, v, fv, dt, dw):
    sx = np.sin(x)

    return x + v*dt, v + fv(x, v, sx)*dt + gv(x, v, sx)*dw

def MilsteinStep(x, v, fv, dt, dw):
    sx = np.sin(x)
    g = gv(x, v, sx)

    return x + v*dt, v + fv(x, v, sx)*dt + g*dw - 0.5*a*b*g*(dw*dw - dt)
    

def error_pendulum_fast(x0, v0, T, fv, N, Scheme, M=2000, ref_factor=5, seed=0):
    rng = np.random.default_rng(seed)

    k = 2**ref_factor
    dt = T / N
    dt_ref = dt / k
    sqrt_dt_ref = np.sqrt(dt_ref)

    xc = np.full(M, x0, dtype=float)
    vc = np.full(M, v0, dtype=float)
    xr = np.full(M, x0, dtype=float)
    vr = np.full(M, v0, dtype=float)

    for _ in range(N):
        dW_block = rng.normal(0.0, sqrt_dt_ref, size=(M, k))
        dWc = dW_block.sum(axis=1)

        for j in range(k):
            dWr = dW_block[:, j]

            xr, vr = Scheme(xr, vr, fv, dt_ref, dWr) 
        
        xc, vc = Scheme(xc, vc, fv, dt, dWc)

    dx_strong = xc - xr
    dv_strong = vc - vr
    dx_weak = np.mean(xc) - np.mean(xr)
    dv_weak = np.mean(vc) - np.mean(vr)
    return float(np.mean(np.sqrt(dx_strong*dx_strong + dv_strong*dv_strong))), float(np.sqrt(dx_weak*dx_weak + dv_weak*dv_weak))


def CreatePlot(scheme,fv,T,fd):
    x0, v0 = 1.0, 0.0
    M = 2000
    ref_factor = 4
    seed = 500

    Ns = np.unique(
        np.round(2**np.linspace(10, 15, 50)).astype(int)
    )  
    dts = T / Ns
    from joblib import Parallel, delayed
    def run_one(N, dt, idx):
        strong, weak = error_pendulum_fast(
            x0, v0, T, fv,
            N,scheme, M=M, ref_factor=ref_factor, seed=seed + idx
        )
        print(f"N={N:6d}, dt={dt:.3e}, strong={strong:.3e} weak={weak:.3e}")
        return strong, weak
    
    errs = Parallel(n_jobs=-1)(
        delayed(run_one)(N, dt, i)
        for i, (N, dt) in enumerate(zip(Ns, dts))
    )
    errs = np.array(errs)

    errsStrong = errs[:,0]
    errsWeak = errs[:,1]
    refStrong = errsStrong[0] * (dts / dts[0])**0.5
    refweak = errsWeak[0] * (dts / dts[0])**1.0

    plt.figure(figsize=(7.5, 5.0))
    plt.loglog(dts, errsStrong, marker="o", linewidth=2, label="Strong error (RMS)")
    plt.loglog(dts, errsWeak, marker="o", linewidth=2, label="Weak error (RMS)")
    plt.loglog(dts, refStrong, linestyle="--", label=r"reference slope $1/2$")
    plt.loglog(dts, refweak, linestyle="--", label=r"reference slope $1$", color='red')
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("dt")
    plt.ylabel("error")
    # plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fd)




   

if __name__ == "__main__":
    CreatePlot(EulerMaruyamaStep,itofv,1.0,"Plot1.png")
    CreatePlot(EulerMaruyamaStep,stratfv,1.0,"Plot2.png")

