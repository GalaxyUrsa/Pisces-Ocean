"""
FastAPI backend entry point.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import os
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .data import load_sound_speed, load_from_path
from .figures import make_layer_fig, make_profile_fig, make_transect_fig, make_volume_fig, VAR_META
from .config import load_config, save_config

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_state: dict = {}

VARIABLES = {"ss", "temp", "salt"}


def _is_ready():
    return "ss" in _state


def _require_data():
    if not _is_ready():
        raise HTTPException(status_code=503, detail="No data loaded. Please upload a .nc file first.")


def _get_data(variable: str):
    """Return the numpy array for the requested variable."""
    if variable == "temp":
        return _state["temp"]
    if variable == "salt":
        return _state["salt"]
    return _state["ss"]


def _init_state(ss, temp, salt, lats, lons, depths):
    _state["ss"]     = ss
    _state["temp"]   = temp
    _state["salt"]   = salt
    _state["lats"]   = lats
    _state["lons"]   = lons
    _state["depths"] = depths
    for var in ("ss", "temp", "salt"):
        _state[f"{var}_min"] = float(VAR_META[var]["vmin"])
        _state[f"{var}_max"] = float(VAR_META[var]["vmax"])
    # Clear any previously cached volume figures
    for key in ("volume_ss", "volume_temp", "volume_salt"):
        _state.pop(key, None)
    print("Data loaded. Volume figures will be computed on first request.")


def _ensure_volume(variable: str):
    key = f"volume_{variable}"
    if key not in _state:
        data = _get_data(variable)
        vmin = _state.get(f"{variable}_min")
        vmax = _state.get(f"{variable}_max")
        _state[key] = make_volume_fig(data, _state["lats"], _state["lons"],
                                      _state["depths"], variable=variable,
                                      vmin=vmin, vmax=vmax)
    return _state[key]


def _var_range(variable: str):
    return _state.get(f"{variable}_min"), _state.get(f"{variable}_max")


def init_data(nc_dir: str, date_str: str):
    ss, temp, salt, lats, lons, depths = load_sound_speed(nc_dir, date_str)
    _init_state(ss, temp, salt, lats, lons, depths)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Ocean Sound Speed API")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Point(BaseModel):
    lat: float
    lon: float


class ProfileRequest(BaseModel):
    lat: float
    lon: float
    depth_idx: int
    variable: str = "ss"
    depth_range: Optional[list] = None
    value_range: Optional[list] = None


class TransectRequest(BaseModel):
    p1: Point
    p2: Point
    depth_idx: int
    variable: str = "ss"
    depth_range: Optional[list] = None
    value_range: Optional[list] = None


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/api/status")
def get_status():
    return {"ready": _is_ready()}


@app.get("/api/config")
def get_config():
    return load_config()


@app.post("/api/config")
async def post_config(request: Request):
    cfg = await request.json()
    save_config(cfg)
    return cfg


@app.post("/api/upload")
async def upload_nc(file: UploadFile = File(...)):
    if not file.filename.endswith(".nc"):
        raise HTTPException(status_code=400, detail="Only .nc files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        ss, temp, salt, lats, lons, depths = load_from_path(tmp_path)
        _init_state(ss, temp, salt, lats, lons, depths)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to load file: {e}")
    finally:
        os.unlink(tmp_path)

    return {
        "ok": True,
        "filename": file.filename,
        "shape": list(ss.shape),
        "ss_min": _state["ss_min"],
        "ss_max": _state["ss_max"],
    }


@app.get("/api/meta")
def get_meta():
    _require_data()
    depths = _state["depths"]
    lats   = _state["lats"]
    lons   = _state["lons"]
    return {
        "depths":      [float(d) for d in depths],
        "lat_range":   [float(lats[0]), float(lats[-1])],
        "lon_range":   [float(lons[0]), float(lons[-1])],
        "variables": {
            "ss":   {"min": _state["ss_min"],   "max": _state["ss_max"]},
            "temp": {"min": _state["temp_min"], "max": _state["temp_max"]},
            "salt": {"min": _state["salt_min"], "max": _state["salt_max"]},
        },
        "grid_shape":  [int(lats.shape[0]), int(lons.shape[0])],
    }


@app.get("/api/volume")
def get_volume(variable: Annotated[str, Query()] = "ss",
               cmin: Annotated[Optional[float], Query()] = None,
               cmax: Annotated[Optional[float], Query()] = None,
               colorscale: Annotated[Optional[str], Query()] = None,
               color_min: Annotated[Optional[str], Query()] = None,
               color_max: Annotated[Optional[str], Query()] = None):
    _require_data()
    if variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"variable must be one of {VARIABLES}")
    if cmin is None and cmax is None and colorscale is None and color_min is None and color_max is None:
        return _ensure_volume(variable)
    data   = _get_data(variable)
    custom = [color_min, color_max] if (color_min and color_max) else None
    return make_volume_fig(data, _state["lats"], _state["lons"],
                           _state["depths"], variable=variable,
                           vmin=cmin, vmax=cmax,
                           colorscale=None if custom else colorscale,
                           colorscale_custom=custom)


@app.get("/api/layer/{depth_idx}")
def get_layer(depth_idx: int,
              variable: Annotated[str, Query()] = "ss",
              points:   Annotated[Optional[str], Query()] = None,
              cmin:     Annotated[Optional[float], Query()] = None,
              cmax:     Annotated[Optional[float], Query()] = None,
              colorscale: Annotated[Optional[str], Query()] = None,
              color_min: Annotated[Optional[str], Query()] = None,
              color_max: Annotated[Optional[str], Query()] = None):
    _require_data()
    if variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"variable must be one of {VARIABLES}")
    depths = _state["depths"]
    if depth_idx < 0 or depth_idx >= len(depths):
        raise HTTPException(status_code=400, detail="depth_idx out of range")

    pts = []
    if points:
        try:
            pts = json.loads(points)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid points JSON")

    vmin   = cmin if cmin is not None else _state.get(f"{variable}_min")
    vmax   = cmax if cmax is not None else _state.get(f"{variable}_max")
    data   = _get_data(variable)
    custom = [color_min, color_max] if (color_min and color_max) else None
    fig    = make_layer_fig(data, _state["lats"], _state["lons"],
                            depths, depth_idx, pts, variable=variable,
                            vmin=vmin, vmax=vmax,
                            colorscale=None if custom else colorscale,
                            colorscale_custom=custom)
    depth_val = float(depths[depth_idx])
    n_depth   = len(depths)
    var_labels = {"ss": "声速", "temp": "温度", "salt": "盐度"}
    return {
        "figure": fig,
        "title":  f"{var_labels[variable]}  {depth_val:.1f} m（第 {depth_idx + 1}/{n_depth} 层）",
    }


@app.post("/api/profile")
def get_profile(req: ProfileRequest):
    _require_data()
    if req.variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"variable must be one of {VARIABLES}")
    depths = _state["depths"]
    if req.depth_idx < 0 or req.depth_idx >= len(depths):
        raise HTTPException(status_code=400, detail="depth_idx out of range")

    data = _get_data(req.variable)
    fig, title, info = make_profile_fig(
        data, _state["lats"], _state["lons"], depths,
        req.lat, req.lon, req.depth_idx,
        variable=req.variable,
        depth_range=tuple(req.depth_range) if req.depth_range else None,
        value_range=tuple(req.value_range) if req.value_range else None,
    )
    return {"figure": fig, "title": title, "info": info}


@app.post("/api/transect")
def get_transect(req: TransectRequest):
    _require_data()
    if req.variable not in VARIABLES:
        raise HTTPException(status_code=400, detail=f"variable must be one of {VARIABLES}")
    depths = _state["depths"]
    if req.depth_idx < 0 or req.depth_idx >= len(depths):
        raise HTTPException(status_code=400, detail="depth_idx out of range")

    data = _get_data(req.variable)
    fig, title, info = make_transect_fig(
        data, _state["lats"], _state["lons"], depths,
        req.p1.model_dump(), req.p2.model_dump(), req.depth_idx,
        variable=req.variable,
        depth_range=tuple(req.depth_range) if req.depth_range else None,
        value_range=tuple(req.value_range) if req.value_range else None,
    )
    return {"figure": fig, "title": title, "info": info}


# ---------------------------------------------------------------------------
# Static files + index
# ---------------------------------------------------------------------------

def _get_frontend_dir():
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, "frontend")
    return os.path.join(os.path.dirname(__file__), "..", "frontend")


_FRONTEND = _get_frontend_dir()

app.mount("/static", StaticFiles(directory=_FRONTEND), name="static")


@app.middleware("http")
async def no_cache_static(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def index():
    return FileResponse(os.path.join(_FRONTEND, "index.html"),
                        headers={"Cache-Control": "no-store"})
