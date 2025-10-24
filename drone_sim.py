import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle
from pathlib import Path

def simulate_1d_motion(accel, v0=0.0, x0=0.0, dt=0.02, t_max=10.0):
    t = np.arange(0.0, t_max + 1e-9, dt)
    n = t.size
    if np.isscalar(accel):
        a = np.full(n, float(accel))
    elif callable(accel):
        a = np.array([float(accel(tt)) for tt in t])
    else:
        arr = np.asarray(accel, dtype=float)
        if arr.size == n:
            a = arr.copy()
        else:
            old_t = np.linspace(0, t_max, arr.size)
            a = np.interp(t, old_t, arr)
    v = np.empty(n)
    x = np.empty(n)
    v[0] = v0
    x[0] = x0
    for i in range(n - 1):
        x[i+1] = x[i] + v[i]*dt + 0.5 * a[i] * dt**2
        v[i+1] = v[i] + a[i] * dt
    return t, x, v, a

def make_animation(t, x, v, a, filename="/mnt/data/1d_motion.gif", fps=30, x_margin=0.1):
    fig, ax = plt.subplots(figsize=(8, 2.2))
    ax.set_ylim(-1, 1)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    xrange = xmax - xmin if xmax != xmin else 1.0
    ax.set_xlim(xmin - x_margin * xrange, xmax + x_margin * xrange)
    ax.get_yaxis().set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.hlines(0, xmin - x_margin * xrange, xmax + x_margin * xrange, linewidth=2)
    dot = Circle((x[0], 0), 0.03 * xrange, ec='k', fill=True)
    ax.add_patch(dot)
    trail, = ax.plot([], [], linewidth=1)
    time_text = ax.text(0.01, 0.85, "", transform=ax.transAxes)
    pos_text  = ax.text(0.01, 0.70, "", transform=ax.transAxes)
    vel_text  = ax.text(0.01, 0.55, "", transform=ax.transAxes)
    acc_text  = ax.text(0.01, 0.40, "", transform=ax.transAxes)

    def init():
        trail.set_data([], [])
        dot.center = (x[0], 0)
        time_text.set_text("")
        pos_text.set_text("")
        vel_text.set_text("")
        acc_text.set_text("")
        return trail, dot, time_text, pos_text, vel_text, acc_text

    n_frames = t.size
    def animate(i):
        xi = x[i]
        dot.center = (xi, 0)
        trail.set_data(x[:i+1], np.zeros(i+1))
        time_text.set_text(f"t = {t[i]:.2f} s")
        pos_text.set_text(f"x = {x[i]:.3f} m")
        vel_text.set_text(f"v = {v[i]:.3f} m/s")
        acc_text.set_text(f"a = {a[i]:.3f} m/s²")
        return trail, dot, time_text, pos_text, vel_text, acc_text

    interval = 1000.0 * (t[1] - t[0])
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                                   interval=interval, blit=True)
    writer = animation.PillowWriter(fps=fps)
    out_path = Path(filename)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    return str(out_path)

# Example to create one GIF:
if __name__ == "__main__":
    accel_const = 4
    t, x, v, a = simulate_1d_motion(accel_const, v0=0.0, x0=0.0, dt=0.02, t_max=8.0)
    gif_path = make_animation(t, x, v, a, filename="/mnt/data/1d_motion_constant.gif", fps=50)
    print("Saved constant-accel animation to:", gif_path)
