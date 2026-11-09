"""Approved Kansas branching-plane construction; coordinates metres, depth positive down."""

import numpy as np

import matplotlib.pyplot as plt

INJ = (5, 10)

PROD = (9, 16)

INJ_OPEN = (8, 12)

PROD_OPEN = (1, 4)

NOTCH = (14, 11)

NZ = 12

DZ = 5.0

VE = 40.0

MAIN_COLOR = "#C2185B"

SPLAY_COLOR = "#E040A0"

MAIN_NW = np.array([-3800.0, 4300.0])

MAIN_SE = np.array([3000.0, -800.0])

SPLAY_NW = np.array([-2700.0, 3200.0])

SPLAY_SE = np.array([2900.0, -3800.0])

def interp_top(xs, ys, top, x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    scalar = np.ndim(x) == 0 and np.ndim(y) == 0
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    ix = np.interp(x, xs, np.arange(len(xs)))
    iy = np.interp(y, ys, np.arange(len(ys)))
    i0 = np.clip(np.floor(ix).astype(int), 0, len(xs) - 2)
    j0 = np.clip(np.floor(iy).astype(int), 0, len(ys) - 2)
    fx = ix - i0
    fy = iy - j0
    z = (
        top[j0, i0] * (1 - fx) * (1 - fy)
        + top[j0, i0 + 1] * fx * (1 - fy)
        + top[j0 + 1, i0] * (1 - fx) * fy
        + top[j0 + 1, i0 + 1] * fx * fy
    )
    return float(z[0]) if scalar else z

def cell_center(xs, ys, ij):
    i, j = ij
    return xs[i] + 0.5 * (xs[i + 1] - xs[i]), ys[j] + 0.5 * (ys[j + 1] - ys[j])

def line_intersect(a1, a2, b1, b2):
    da = a2 - a1
    db = b2 - b1
    det = da[0] * db[1] - da[1] * db[0]
    if abs(det) < 1e-9:
        raise ValueError("map traces are parallel")
    t = ((b1[0] - a1[0]) * db[1] - (b1[1] - a1[1]) * db[0]) / det
    return a1 + t * da, t

def line_section_intersect(p1, p2, o, u):
    """Intersection of 2-D line p1-p2 with section origin o + s u, |u|=1. Returns (xy, s)."""
    dp = p2 - p1
    # p1 + t dp = o + s u
    A = np.array([[dp[0], -u[0]], [dp[1], -u[1]]], float)
    t, s = np.linalg.solve(A, o - p1)
    return p1 + t * dp, float(s)

def plane_from_points(p1, p2, p3):
    n = np.cross(p2 - p1, p3 - p1)
    n = n / np.linalg.norm(n)
    # depth-down: a downward normal makes n[2] > 0 useful for z = ...
    if n[2] < 0:
        n = -n
    return dict(p0=p1.copy(), n=n)

def plane_z(pl, x, y):
    n, p0 = pl["n"], pl["p0"]
    return p0[2] - (n[0] * (x - p0[0]) + n[1] * (y - p0[1])) / n[2]

def plane_dip(pl):
    n = pl["n"]
    # n · (x,y,z) form; slope of z(x,y) is -(n[0], n[1])/n[2]
    gx, gy = -n[0] / n[2], -n[1] / n[2]
    slope = float(np.hypot(gx, gy))
    dip = float(np.degrees(np.arctan(slope)))
    # dip direction (down-dip azimuth from north, clockwise)
    az = float((np.degrees(np.arctan2(gx, gy)) + 360.0) % 360.0)
    strike = (az - 90.0) % 180.0
    return dip, az, strike, slope

def inside_l(xs, ys, x, y):
    x_notch = xs[NOTCH[0]]
    y_notch = ys[NOTCH[1] + 1]
    in_box = (xs[0] <= x) & (x <= xs[-1]) & (ys[0] <= y) & (y <= ys[-1])
    in_notch = (x >= x_notch) & (y <= y_notch)
    return in_box & ~in_notch

def top_trace(xs, ys, top, pl, clip=None, n=700):
    """Contour where the plane meets the structural top."""
    gx = np.linspace(xs[0], xs[-1], n)
    gy = np.linspace(ys[0], ys[-1], n)
    XX, YY = np.meshgrid(gx, gy)
    ZZ = plane_z(pl, XX, YY)
    TT = interp_top(xs, ys, top, XX, YY)
    fig = plt.figure()
    cs = plt.contour(XX, YY, ZZ - TT, levels=[0.0])
    segs = []
    for c in cs.allsegs[0]:
        segs.append(c)
    plt.close(fig)
    if not segs:
        return np.empty((0, 2))
    # pick the longest segment, then optionally clip
    segs = sorted(segs, key=len, reverse=True)
    xy = segs[0]
    keep = inside_l(xs, ys, xy[:, 0], xy[:, 1])
    xy = xy[keep]
    if clip is not None:
        xy = xy[clip(xy)]
    return xy

def construct(xs, ys, top):
    x_inj, y_inj = cell_center(xs, ys, INJ)
    x_prod, y_prod = cell_center(xs, ys, PROD)
    length = float(np.hypot(x_prod - x_inj, y_prod - y_inj))
    u = np.array([x_prod - x_inj, y_prod - y_inj]) / length
    o = np.array([x_inj, y_inj])
    B_xy, tB = line_intersect(MAIN_NW, MAIN_SE, SPLAY_NW, SPLAY_SE)
    N_xy, sN = line_section_intersect(MAIN_NW, MAIN_SE, o, u)
    S_xy, sS = line_section_intersect(SPLAY_NW, SPLAY_SE, o, u)
    zB = interp_top(xs, ys, top, B_xy[0], B_xy[1])
    zN = interp_top(xs, ys, top, N_xy[0], N_xy[1])
    zS = interp_top(xs, ys, top, S_xy[0], S_xy[1])
    s_stem = sN - 960.0
    stem_xy = o + s_stem * u
    z_stem = interp_top(xs, ys, top, stem_xy[0], stem_xy[1]) + NZ * DZ - 2.0
    B = np.array([B_xy[0], B_xy[1], zB])
    N = np.array([N_xy[0], N_xy[1], zN])
    S = np.array([S_xy[0], S_xy[1], zS])
    STEM = np.array([stem_xy[0], stem_xy[1], z_stem])
    main_pl = plane_from_points(B, N, STEM)
    sJ = 1700.0
    J_xy = o + sJ * u
    zJ = float(plane_z(main_pl, J_xy[0], J_xy[1]))
    J = np.array([J_xy[0], J_xy[1], zJ])
    splay_pl = plane_from_points(B, S, J)
    return dict(main=main_pl, splay=splay_pl, B=B, N=N, S=S, STEM=STEM, J=J, o=o, u=u, length=length)
