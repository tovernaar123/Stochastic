import time
import numpy as np
import matplotlib.pyplot as plt

def angle_diff(x1, x2):
    d = x1 - x2
    return (d + np.pi) % (2*np.pi) - np.pi

def strong_error_pendulum_fast(
    x0, v0, T, a, b, c,
    N, M=2000, ref_factor=5, seed=0
):
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

            sx = np.sin(xr)
            sin2x = np.sin(2*xr)

            dx = vr
            dv = -b*vr - sx + c*sin2x
            gv = -a*(b*vr + sx)

            xr = xr + dx * dt_ref
            vr = vr + dv * dt_ref + gv * dWr
        sx = np.sin(xc)
        cx = np.cos(xc)
        sin2x = 2.0 * sx * cx

        dx = vc
        dv = -b*vc - sx + c*sin2x
        gv = -a*(b*vc + sx)

        xc = xc + dx * dt
        vc = vc + dv * dt + gv * dWc

    dx = angle_diff(xc, xr)
    dv = vc - vr
    return float(np.mean(np.sqrt(dx*dx + dv*dv)))

def convergence_plot_strong():
    x0, v0 = 1.0, 0.0
    T = 2.0
    a, b, c = 0.3, 0.6, 0.3

    M = 100
    ref_factor = 7
    seed = 123

    Ns = np.unique(
        np.round(2**np.linspace(11, 16, 50)).astype(int)
    )  
    dts = T / Ns

    from joblib import Parallel, delayed

    def run_one(N, dt, idx):
        e = strong_error_pendulum_fast(
            x0, v0, T, a, b, c,
            N, M=M, ref_factor=ref_factor, seed=seed + idx
        )
        print(f"N={N:6d}, dt={dt:.3e}, strong_RMS={e:.3e}")
        return e
    start = time.time()
    errs = Parallel(n_jobs=-1)(
        delayed(run_one)(N, dt, i)
        for i, (N, dt) in enumerate(zip(Ns, dts))
    )
    end = time.time()
    print(end - start)
    print("heelloooooo")
    errs = np.array(errs)

    ref = errs[0] * (dts / dts[0])**0.5

    plt.figure(figsize=(7.5, 5.0))
    plt.loglog(dts, errs, marker="o", linewidth=2, label="Strong error (RMS)")
    plt.loglog(dts, ref, linestyle="--", label=r"reference slope $1/2$")
    plt.gca().invert_xaxis()
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("dt")
    plt.ylabel("error")
    plt.title(f"Strong error, M={M}, ref_factor={ref_factor}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("test.png")

if __name__ == "__main__":
    convergence_plot_strong()
