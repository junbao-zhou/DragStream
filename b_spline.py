# Click to place control points, then see a clamped B-spline plus
# equally spaced points along the curve.
#
# Controls:
# - Left-click: add a control point
# - Right-click: remove the last control point
# - '+' / '-': increase/decrease the number of equally spaced points
# - 'c': clear all points
#
# Requirements: numpy, matplotlib, scipy

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.integrate import cumulative_trapezoid

# ------------- Spline utilities -------------


def _open_uniform_knots(n, k):
    # n = number of control points, k = degree
    m = n + k + 1
    t = np.zeros(m, dtype=float)
    t[-(k + 1) :] = 1.0
    num_interior = n - k - 1
    if num_interior > 0:
        t[k + 1 : n] = np.linspace(
            1 / (n - k), (n - k - 1) / (n - k), num_interior
        )
    return t


def build_clamped_bspline(control_points, degree=3):
    """
    Build an open-uniform (clamped) B-spline that passes through the
    first and last control points.
    """
    P = np.asarray(control_points, dtype=float)
    n = len(P)
    if n < 2:
        raise ValueError("Need at least 2 control points")
    k = min(degree, n - 1)  # degree cannot exceed n-1
    t = _open_uniform_knots(n, k)
    return BSpline(t, P, k, axis=0)


def equidistant_points_on_spline(spline, num_points, grid=6000):
    """
    Return `num_points` points equally spaced in arc length along `spline`.
    """
    if num_points < 2:
        raise ValueError("num_points must be >= 2")

    u = np.linspace(0.0, 1.0, grid)
    dCdu = spline.derivative()(u)
    speed = np.linalg.norm(dCdu, axis=1)

    s = cumulative_trapezoid(speed, u, initial=0.0)
    total_len = s[-1]

    if total_len <= 1e-12:
        P0 = spline(0.0)
        return np.repeat(P0[None, :], num_points, axis=0)

    s_targets = np.linspace(0.0, total_len, num_points)
    u_targets = np.interp(s_targets, s, u)
    return spline(u_targets)


# ------------- Interactive demo -------------

if __name__ == "__main__":
    # Optional: draw over an image. Uncomment and set path if needed.
    # img = plt.imread("your_image.png")
    # H, W = img.shape[:2]

    points = []
    sample_count = [25]  # use a list so we can modify inside callbacks
    sample_count = [2]  # use a list so we can modify inside callbacks

    fig, ax = plt.subplots()
    ax.set_aspect("equal", adjustable="box")

    # If drawing over an image, uncomment:
    # ax.imshow(img, extent=[0, W, H, 0], origin='upper')
    # ax.set_xlim(0, W)
    # ax.set_ylim(H, 0)
    # Otherwise, use a unit square canvas:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    title_template = (
        "Left-click: add, Right-click: undo | +/-: change N | c: clear | N = {}"
    )
    ax.set_title(title_template.format(sample_count[0]))

    # Artists
    scatter_ctrl = ax.scatter(
        [], [], c="r", s=25, zorder=3, label="control points"
    )
    (ctrl_line,) = ax.plot(
        [], [], "r--", lw=1, alpha=0.6, zorder=2, label="control polygon"
    )
    (curve_line,) = ax.plot([], [], "g-", lw=2, zorder=1, label="B-spline")
    eq_scatter = ax.scatter(
        [], [], c="b", s=20, zorder=4, label="equally spaced points"
    )

    ax.legend(loc="upper right")

    def update_plot():
        if points:
            P = np.array(points, dtype=float)
            scatter_ctrl.set_offsets(P)
            ctrl_line.set_data(P[:, 0], P[:, 1])
        else:
            scatter_ctrl.set_offsets(np.empty((0, 2)))
            ctrl_line.set_data([], [])

        if len(points) >= 2:
            try:
                spl = build_clamped_bspline(points, degree=3)
                # For a smooth preview of the curve
                C = spl(np.linspace(0, 1, 1000))
                curve_line.set_data(C[:, 0], C[:, 1])

                # Equally spaced points along the curve
                N = max(2, sample_count[0])
                eq_pts = equidistant_points_on_spline(
                    spl, num_points=N, grid=8000
                )
                eq_scatter.set_offsets(eq_pts)
            except Exception as e:
                print("Error building spline:", e)
                curve_line.set_data([], [])
                eq_scatter.set_offsets(np.empty((0, 2)))
        else:
            curve_line.set_data([], [])
            eq_scatter.set_offsets(np.empty((0, 2)))

        ax.set_title(title_template.format(sample_count[0]))
        fig.canvas.draw_idle()

    def onclick(event):
        if event.inaxes != ax:
            return
        if event.button == 1:  # left: add
            if event.xdata is None or event.ydata is None:
                return
            points.append([event.xdata, event.ydata])
            update_plot()
        elif event.button == 3:  # right: undo last
            if points:
                points.pop()
                update_plot()

    def onkey(event):
        if event.key in ["+", "="]:
            sample_count[0] += 1
            update_plot()
        elif event.key == "-":
            if sample_count[0] > 2:
                sample_count[0] -= 1
                update_plot()
        elif event.key in ["c", "C"]:
            points.clear()
            update_plot()

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)

    plt.show()
