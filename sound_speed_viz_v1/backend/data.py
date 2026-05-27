"""
Data loading and sound speed calculation.
"""

import os
import numpy as np
import xarray as xr


def sound_speed_chen_millero(T, S, z):
    """UNESCO / Chen-Millero (1977) sound speed formula."""
    P = z * 0.1  # dbar
    Cw = (
        (1402.388
         + 5.03830 * T - 5.81090e-2 * T**2 + 3.3432e-4 * T**3
         - 1.47797e-6 * T**4 + 3.1419e-9 * T**5)
        + P * (0.153563 + 6.8999e-4 * T - 8.1829e-6 * T**2
               + 1.3632e-7 * T**3 - 6.1260e-10 * T**4)
        + P**2 * (3.1260e-5 - 1.7111e-6 * T + 2.5986e-8 * T**2
                  - 2.5353e-10 * T**3 + 1.0415e-12 * T**4)
        + P**3 * (-9.7729e-9 + 3.8513e-10 * T - 2.3654e-12 * T**2)
    )
    A = (
        (1.389 - 1.262e-2 * T + 7.166e-5 * T**2
         + 2.008e-6 * T**3 - 3.21e-8 * T**4)
        + P * (9.4742e-5 - 1.2583e-5 * T - 6.4928e-8 * T**2
               + 1.0515e-8 * T**3 - 2.0142e-10 * T**4)
        + P**2 * (-3.9064e-7 + 9.1061e-9 * T - 1.6009e-10 * T**2
                  + 7.994e-12 * T**3)
        + P**3 * (1.100e-10 + 6.651e-12 * T - 3.391e-13 * T**2)
    )
    B = -1.922e-2 - 4.42e-5 * T + P * 7.3637e-5 + P * T * 1.7950e-7
    D = 1.727e-3 - 7.9836e-6 * P
    return Cw + A * S + B * S**1.5 + D * S**2


DEPTHS = np.array([
    0.5, 9.6, 18.5, 29.4, 40.3, 55.8, 65.8, 77.9, 92.3, 109.7,
    130.7, 155.9, 186.1, 222.5, 266.0, 318.1, 380.2, 453.9, 541.1, 643.6
])


def _load_nc(ds) -> tuple:
    """Extract T, S, lats, lons, depths from an open xarray Dataset."""
    T    = ds["thetao"].isel(time=0).values
    S    = ds["so"].isel(time=0).values
    lats = ds["latitude"].values
    lons = ds["longitude"].values
    # Use depth coordinate from file if present, otherwise fall back to DEPTHS
    for dim in ("depth", "deptht", "lev", "level", "z_l", "z_t"):
        if dim in ds.coords or dim in ds.dims:
            depths = ds[dim].values.astype(float)
            break
    else:
        depths = DEPTHS
    return T, S, lats, lons, depths


def _compute_ss(T, S, depths):
    z = depths[:, np.newaxis, np.newaxis] * np.ones_like(T)
    return sound_speed_chen_millero(T, S, z)


def load_sound_speed(nc_dir: str, date_str: str):
    pred_path = os.path.join(nc_dir, f"prediction_{date_str}.nc")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Not found: {pred_path}")
    return load_from_path(pred_path)


def load_from_path(path: str):
    """Load from an arbitrary .nc file path (used by upload endpoint)."""
    ds = xr.open_dataset(path)
    T, S, lats, lons, depths = _load_nc(ds)
    ds.close()
    ss = _compute_ss(T, S, depths)
    return ss, T, S, lats, lons, depths


def nearest_idx(arr, val):
    return int(np.argmin(np.abs(arr - val)))
