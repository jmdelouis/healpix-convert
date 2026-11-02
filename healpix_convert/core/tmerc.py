"""Inverse transverse Mercator (e.g. UTM -> lon/lat) in torch, on any device.

The "exact" algorithm of PROJ (Poder/Engsager, Krueger series of order 6,
PROJ ``src/projections/tmerc.cpp``), with PROJ's order of operations, so the
coordinates of projected inputs can be computed on the GPU. On a 10 m
Sentinel-2 grid the results differ from pyproj's by a few units in the last
place (at most 3e-14 degree, a few nanometres), from the math libraries.
"""
import math

import numpy as np
import torch

ORDER = 6


def _coefficients(f, k0, lat0):
    n = f / (2 - f)
    cgb, cbg, utg, gtu = [0.0] * 6, [0.0] * 6, [0.0] * 6, [0.0] * 6
    np_ = n
    cgb[0] = n * (2 + n * (-2 / 3.0 + n * (-2 + n * (116 / 45.0 + n * (26 / 45.0 + n * (-2854 / 675.0))))))
    cbg[0] = n * (-2 + n * (2 / 3.0 + n * (4 / 3.0 + n * (-82 / 45.0 + n * (32 / 45.0 + n * (4642 / 4725.0))))))
    np_ *= n
    cgb[1] = np_ * (7 / 3.0 + n * (-8 / 5.0 + n * (-227 / 45.0 + n * (2704 / 315.0 + n * (2323 / 945.0)))))
    cbg[1] = np_ * (5 / 3.0 + n * (-16 / 15.0 + n * (-13 / 9.0 + n * (904 / 315.0 + n * (-1522 / 945.0)))))
    np_ *= n
    cgb[2] = np_ * (56 / 15.0 + n * (-136 / 35.0 + n * (-1262 / 105.0 + n * (73814 / 2835.0))))
    cbg[2] = np_ * (-26 / 15.0 + n * (34 / 21.0 + n * (8 / 5.0 + n * (-12686 / 2835.0))))
    np_ *= n
    cgb[3] = np_ * (4279 / 630.0 + n * (-332 / 35.0 + n * (-399572 / 14175.0)))
    cbg[3] = np_ * (1237 / 630.0 + n * (-12 / 5.0 + n * (-24832 / 14175.0)))
    np_ *= n
    cgb[4] = np_ * (4174 / 315.0 + n * (-144838 / 6237.0))
    cbg[4] = np_ * (-734 / 315.0 + n * (109598 / 31185.0))
    np_ *= n
    cgb[5] = np_ * (601676 / 22275.0)
    cbg[5] = np_ * (444337 / 155925.0)

    np_ = n * n
    qn = k0 / (1 + n) * (1 + np_ * (1 / 4.0 + np_ * (1 / 64.0 + np_ / 256.0)))
    utg[0] = n * (-0.5 + n * (2 / 3.0 + n * (-37 / 96.0 + n * (1 / 360.0 + n * (81 / 512.0 + n * (-96199 / 604800.0))))))
    gtu[0] = n * (0.5 + n * (-2 / 3.0 + n * (5 / 16.0 + n * (41 / 180.0 + n * (-127 / 288.0 + n * (7891 / 37800.0))))))
    utg[1] = np_ * (-1 / 48.0 + n * (-1 / 15.0 + n * (437 / 1440.0 + n * (-46 / 105.0 + n * (1118711 / 3870720.0)))))
    gtu[1] = np_ * (13 / 48.0 + n * (-3 / 5.0 + n * (557 / 1440.0 + n * (281 / 630.0 + n * (-1983433 / 1935360.0)))))
    np_ *= n
    utg[2] = np_ * (-17 / 480.0 + n * (37 / 840.0 + n * (209 / 4480.0 + n * (-5569 / 90720.0))))
    gtu[2] = np_ * (61 / 240.0 + n * (-103 / 140.0 + n * (15061 / 26880.0 + n * (167603 / 181440.0))))
    np_ *= n
    utg[3] = np_ * (-4397 / 161280.0 + n * (11 / 504.0 + n * (830251 / 7257600.0)))
    gtu[3] = np_ * (49561 / 161280.0 + n * (-179 / 168.0 + n * (6601661 / 7257600.0)))
    np_ *= n
    utg[4] = np_ * (-4583 / 161280.0 + n * (108847 / 3991680.0))
    gtu[4] = np_ * (34729 / 80640.0 + n * (-3418889 / 1995840.0))
    np_ *= n
    utg[5] = np_ * (-20648693 / 638668800.0)
    gtu[5] = np_ * (212378941 / 319334400.0)

    z = _gatg(cbg, lat0, math.cos(2 * lat0), math.sin(2 * lat0))
    zb = -qn * (z + _clens(gtu, 2 * z))
    return cgb, utg, qn, zb


def _gatg(p, b, cos_2b, sin_2b):
    two_cos_2b = 2 * cos_2b
    h, h1, h2 = 0.0, p[-1], 0.0
    for c in reversed(p[:-1]):
        h = -h2 + two_cos_2b * h1 + c
        h2, h1 = h1, h
    return b + h * sin_2b


def _clens(a, arg_r):
    r = 2 * math.cos(arg_r)
    hr, hr1, hr2 = a[-1], 0.0, 0.0
    for c in reversed(a[:-1]):
        hr2, hr1 = hr1, hr
        hr = -hr2 + r * hr1 + c
    return math.sin(arg_r) * hr


def tmerc_inverse(x, y, *, a, f, k0, lon0_deg, lat0_deg, x0, y0, device="cpu"):
    """lon, lat (degrees, float64 numpy) of the projected x, y (metres)."""
    cgb, utg, qn, zb = _coefficients(f, k0, math.radians(lat0_deg))
    xt = (torch.as_tensor(np.asarray(x, dtype="f8"), device=device) - x0) / a
    yt = (torch.as_tensor(np.asarray(y, dtype="f8"), device=device) - y0) / a
    cn = (yt - zb) / qn
    ce = xt / qn
    sin_r, cos_r = torch.sin(2 * cn), torch.cos(2 * cn)
    e2 = torch.exp(2 * ce)
    half_inv = 0.5 / e2
    sinh_i, cosh_i = 0.5 * e2 - half_inv, 0.5 * e2 + half_inv
    # clenS
    r = 2 * cos_r * cosh_i
    i = -2 * sin_r * sinh_i
    hr = torch.full_like(cn, utg[-1])
    hi = torch.zeros_like(cn)
    hr1 = torch.zeros_like(cn)
    hi1 = torch.zeros_like(cn)
    for c in reversed(utg[:-1]):
        hr2, hi2, hr1, hi1 = hr1, hi1, hr, hi
        hr = -hr2 + r * hr1 - i * hi1 + c
        hi = -hi2 + i * hr1 + r * hi1
    rr = sin_r * cosh_i
    ii = cos_r * sinh_i
    dcn = rr * hr - ii * hi
    dce = rr * hi + ii * hr
    cn = cn + dcn
    ce = ce + dce
    sin_cn, cos_cn = torch.sin(cn), torch.cos(cn)
    sinh_ce = torch.sinh(ce)
    lam = torch.atan2(sinh_ce, cos_cn)
    mod = torch.hypot(sinh_ce, cos_cn)
    cn = torch.atan2(sin_cn, mod)
    tmp = 2 * mod / (sinh_ce * sinh_ce + 1)
    sin_2cn = sin_cn * tmp
    cos_2cn = tmp * mod - 1.0
    two_cos = 2 * cos_2cn
    h1 = torch.full_like(cn, cgb[-1])
    h2 = torch.zeros_like(cn)
    h = h1
    for c in reversed(cgb[:-1]):
        h = -h2 + two_cos * h1 + c
        h2, h1 = h1, h
    phi = cn + h * sin_2cn
    # as PROJ: the central meridian added in radians, then to degrees
    lon = (lam + math.radians(lon0_deg)) * (180.0 / math.pi)
    lat = phi * (180.0 / math.pi)
    return lon.cpu().numpy(), lat.cpu().numpy()


def tmerc_params(crs):
    """The tmerc parameters of a pyproj CRS, or None if it is not a
    transverse Mercator projection."""
    op = crs.coordinate_operation
    if op is None or "Transverse Mercator" not in op.method_name:
        return None
    p = {q.name: q.value for q in op.params}
    e = crs.ellipsoid
    return dict(
        a=e.semi_major_metre, f=1.0 / e.inverse_flattening, k0=p["Scale factor at natural origin"],
        lon0_deg=p["Longitude of natural origin"], lat0_deg=p["Latitude of natural origin"],
        x0=p["False easting"], y0=p["False northing"],
    )
