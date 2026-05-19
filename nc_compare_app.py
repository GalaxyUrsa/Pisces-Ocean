"""
NetCDF Comparison Tool
Compare two NetCDF files: variable maps, difference maps, error statistics, depth profiles.

Usage:
    streamlit run nc_compare_app.py
"""

import io
import tempfile
import os
from typing import Optional
import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Theme / CSS
# ---------------------------------------------------------------------------

OCEAN_CSS = """
<style>
/* ── Global background & text ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0d1b2a;
    color: #d6e4f0;
}
[data-testid="stSidebar"] {
    background-color: #112233;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * {
    color: #b8d4e8 !important;
}

/* ── Sidebar section headers ── */
.sidebar-section {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #4a9eca !important;
    margin: 1.2rem 0 0.4rem 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #1e3a5f;
}

/* ── Page title ── */
h1 {
    color: #7ec8e3 !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em;
    margin-bottom: 0.2rem !important;
}

/* ── Tab bar ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 2px solid #1e3a5f;
    gap: 4px;
}
[data-testid="stTabs"] [role="tab"] {
    background: #112233;
    border: 1px solid #1e3a5f;
    border-radius: 6px 6px 0 0;
    color: #7ec8e3 !important;
    font-weight: 600;
    padding: 6px 20px;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #1a3a5c;
    border-bottom: 2px solid #4a9eca;
    color: #ffffff !important;
}

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #112233 0%, #1a3a5c 100%);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.4);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #4a9eca;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #e8f4fd;
    font-family: 'Courier New', monospace;
}
.metric-value.good  { color: #4caf8a; }
.metric-value.warn  { color: #f0c040; }
.metric-value.bad   { color: #e05c5c; }

/* ── Info / error boxes ── */
[data-testid="stAlert"] {
    border-radius: 8px;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    overflow: hidden;
}

/* ── Inputs & selects ── */
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"] {
    background-color: #0d1b2a !important;
    border-color: #1e3a5f !important;
    color: #d6e4f0 !important;
}

/* ── Divider ── */
hr {
    border-color: #1e3a5f !important;
}

/* ── Section label above maps ── */
.section-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4a9eca;
    margin-bottom: 4px;
}
</style>
"""

# Plotly layout defaults (dark ocean theme)
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0d1b2a",
    plot_bgcolor="#112233",
    font=dict(color="#b8d4e8", family="Arial, sans-serif", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_nc_from_path(path: str) -> xr.Dataset:
    return xr.open_dataset(path, engine="netcdf4")


@st.cache_data(show_spinner=False)
def load_nc_from_bytes(data: bytes, filename: str) -> xr.Dataset:
    suffix = os.path.splitext(filename)[-1] or ".nc"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        return xr.open_dataset(tmp.name, engine="netcdf4")
    except Exception:
        os.unlink(tmp.name)
        raise


def detect_variables(ds: xr.Dataset) -> list:
    """Return data variables that have at least 2 spatial dimensions."""
    lat_names = {"lat", "latitude", "y", "nav_lat"}
    lon_names = {"lon", "longitude", "x", "nav_lon"}
    result = []
    for name, var in ds.data_vars.items():
        dims_lower = {d.lower() for d in var.dims}
        has_lat = bool(dims_lower & lat_names)
        has_lon = bool(dims_lower & lon_names)
        if has_lat and has_lon:
            result.append(name)
    return result if result else list(ds.data_vars.keys())


def get_depth_dim(ds: xr.Dataset, var: str) -> Optional[str]:
    """Return the name of the depth dimension, or None if not present."""
    depth_names = {"depth", "lev", "level", "z", "deptht", "depthu", "depthv", "depthw"}
    for dim in ds[var].dims:
        if dim.lower() in depth_names:
            return dim
    return None


def get_depth_coords(ds: xr.Dataset, var: str) -> Optional[np.ndarray]:
    """Return depth coordinate values as a numpy array, or None."""
    dim = get_depth_dim(ds, var)
    if dim is None:
        return None
    if dim in ds.coords:
        return ds.coords[dim].values.astype(float)
    return np.arange(ds[var].sizes[dim], dtype=float)


def get_time_size(ds: xr.Dataset, var: str) -> int:
    """Return the size of the time dimension, or 0 if no time dim."""
    for dim in ds[var].dims:
        if dim.lower() == "time":
            return ds[var].sizes[dim]
    return 0


def extract_slice(ds: xr.Dataset, var: str, time_idx: int) -> np.ndarray:
    """
    Extract a (depth, lat, lon) or (lat, lon) array for the given time index.
    Returns float32 numpy array with NaN for masked/fill values.
    """
    da = ds[var]
    for dim in da.dims:
        if dim.lower() == "time":
            da = da.isel({dim: time_idx})
            break
    arr = da.values.astype(np.float32)
    fill = ds[var].attrs.get("_FillValue", None) or ds[var].attrs.get("missing_value", None)
    if fill is not None:
        arr[arr == fill] = np.nan
    arr[arr < -1e10] = np.nan
    return arr


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute RMSE, MAE, Bias, Pearson r on valid (non-NaN) pixels."""
    valid = ~np.isnan(a) & ~np.isnan(b)
    n = int(valid.sum())
    if n < 2:
        return {"rmse": np.nan, "mae": np.nan, "bias": np.nan, "pearson_r": np.nan, "n_valid": n}
    av, bv = a[valid], b[valid]
    diff = av - bv
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    r, _ = pearsonr(av.ravel(), bv.ravel())
    return {"rmse": rmse, "mae": mae, "bias": bias, "pearson_r": float(r), "n_valid": n}


def compute_all_depth_stats(a3d: np.ndarray, b3d: np.ndarray,
                             depth_values: np.ndarray) -> pd.DataFrame:
    """Compute per-depth statistics. a3d/b3d shape: (depth, lat, lon)."""
    rows = []
    for i in range(a3d.shape[0]):
        s = compute_stats(a3d[i], b3d[i])
        depth_m = float(depth_values[i]) if depth_values is not None else float(i)
        rows.append({"depth_m": depth_m, **s})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _vrange(arr: np.ndarray, pct_lo: float, pct_hi: float):
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return 0.0, 1.0
    return float(np.percentile(valid, pct_lo)), float(np.percentile(valid, pct_hi))


def render_maps(a: np.ndarray, b: np.ndarray,
                label_a: str, label_b: str,
                cmap: str, pct_clip: tuple,
                manual_vmin=None, manual_vmax=None,
                manual_dmax=None,
                lons=None, lats=None) -> None:
    """Render 3-panel interactive Plotly figure: A | B | (A - B)."""
    diff = a - b

    if manual_vmin is not None and manual_vmax is not None:
        vmin, vmax = manual_vmin, manual_vmax
    else:
        vmin, vmax = _vrange(np.concatenate([a.ravel(), b.ravel()]), pct_clip[0], pct_clip[1])

    if manual_dmax is not None:
        dmax = max(manual_dmax, 1e-6)
    else:
        diff_valid = diff[~np.isnan(diff)]
        dmax = float(np.percentile(np.abs(diff_valid), 98)) if len(diff_valid) > 0 else 1.0
        dmax = max(dmax, 1e-6)

    nrows, ncols = a.shape
    if lons is not None and lats is not None:
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        def make_hover(arr):
            texts = []
            for i in range(nrows):
                row = []
                for j in range(ncols):
                    row.append(
                        f"lon: {lon_grid[i,j]:.3f}<br>lat: {lat_grid[i,j]:.3f}<br>"
                        f"A: {a[i,j]:.4f}<br>B: {b[i,j]:.4f}<br>Diff: {diff[i,j]:.4f}"
                    )
                texts.append(row)
            return texts
    else:
        def make_hover(arr):
            texts = []
            for i in range(nrows):
                row = []
                for j in range(ncols):
                    row.append(
                        f"row: {i}, col: {j}<br>"
                        f"A: {a[i,j]:.4f}<br>B: {b[i,j]:.4f}<br>Diff: {diff[i,j]:.4f}"
                    )
                texts.append(row)
            return texts

    hover_text = make_hover(a)

    cmap_map = {
        "RdYlBu_r": "RdYlBu_r", "viridis": "Viridis", "turbo": "Turbo",
        "plasma": "Plasma", "coolwarm": "RdBu",
    }
    plotly_cmap = cmap_map.get(cmap, cmap)

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[label_a, label_b, f"{label_a} − {label_b}"],
        horizontal_spacing=0.06,
    )

    common = dict(hoverinfo="text", hovertemplate="%{text}<extra></extra>")

    fig.add_trace(go.Heatmap(z=a, colorscale=plotly_cmap, zmin=vmin, zmax=vmax,
                              text=hover_text, showscale=True,
                              colorbar=dict(x=0.30, len=0.85, thickness=10,
                                            tickfont=dict(color="#b8d4e8", size=10),
                                            outlinecolor="#1e3a5f"),
                              **common), row=1, col=1)

    fig.add_trace(go.Heatmap(z=b, colorscale=plotly_cmap, zmin=vmin, zmax=vmax,
                              text=hover_text, showscale=True,
                              colorbar=dict(x=0.64, len=0.85, thickness=10,
                                            tickfont=dict(color="#b8d4e8", size=10),
                                            outlinecolor="#1e3a5f"),
                              **common), row=1, col=2)

    fig.add_trace(go.Heatmap(z=diff, colorscale="RdBu_r", zmin=-dmax, zmax=dmax,
                              text=hover_text, showscale=True,
                              colorbar=dict(x=0.99, len=0.85, thickness=10,
                                            tickfont=dict(color="#b8d4e8", size=10),
                                            outlinecolor="#1e3a5f"),
                              **common), row=1, col=3)

    fig.update_layout(
        height=440,
        **PLOTLY_LAYOUT,
    )
    fig.update_annotations(font=dict(color="#7ec8e3", size=13, family="Arial"))
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False,
                     showline=True, linecolor="#1e3a5f")
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False,
                     showline=True, linecolor="#1e3a5f")

    st.plotly_chart(fig, use_container_width=True)


def _metric_color_class(key: str, val: float) -> str:
    """Assign a CSS class based on metric value quality."""
    if np.isnan(val):
        return ""
    if key == "pearson_r":
        if val >= 0.95:
            return "good"
        if val >= 0.80:
            return "warn"
        return "bad"
    return ""


def render_stats_bar(stats: dict) -> None:
    labels = ["RMSE", "MAE", "Bias", "Pearson r"]
    keys   = ["rmse", "mae", "bias", "pearson_r"]
    cols = st.columns(4)
    for col, label, key in zip(cols, labels, keys):
        val = stats[key]
        display = f"{val:.4f}" if not np.isnan(val) else "N/A"
        css_class = _metric_color_class(key, val)
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value {css_class}">{display}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_depth_profile(df: pd.DataFrame, metric: str) -> None:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df[metric], y=df["depth_m"],
        mode="lines+markers",
        line=dict(color="#4a9eca", width=2),
        marker=dict(size=5, color="#7ec8e3",
                    line=dict(color="#0d1b2a", width=1)),
        name=metric.upper(),
        fill="tozerox",
        fillcolor="rgba(74,158,202,0.12)",
    ))
    layout = {**PLOTLY_LAYOUT, "margin": dict(l=60, r=20, t=30, b=50)}
    fig.update_layout(
        xaxis_title=metric.upper(),
        yaxis_title="Depth (m)",
        yaxis=dict(autorange="reversed",
                   gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
        xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f"),
        height=500,
        **layout,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar: file + variable loading
# ---------------------------------------------------------------------------

def sidebar_file_panel(label: str, key_prefix: str):
    """Render sidebar panel for one file. Returns (ds, var, display_label) or (None, None, None)."""
    st.sidebar.markdown(f'<div class="sidebar-section">{label}</div>', unsafe_allow_html=True)
    path_input = st.sidebar.text_input("File path", key=f"{key_prefix}_path",
                                        placeholder="D:/datasets/file.nc")
    uploaded = st.sidebar.file_uploader("Or upload file", type=["nc", "nc4"],
                                         key=f"{key_prefix}_upload")
    display_label = st.sidebar.text_input("Display label", value=label, key=f"{key_prefix}_label")

    ds = None

    if path_input.strip():
        try:
            with st.spinner(f"Loading {label}..."):
                ds = load_nc_from_path(path_input.strip())
        except Exception as e:
            st.sidebar.error(f"Cannot open: {e}")
    elif uploaded is not None:
        try:
            with st.spinner(f"Loading {label}..."):
                ds = load_nc_from_bytes(uploaded.read(), uploaded.name)
        except Exception as e:
            st.sidebar.error(f"Cannot open: {e}")

    if ds is None:
        return None, None, display_label

    vars_list = detect_variables(ds)
    if not vars_list:
        st.sidebar.warning("No spatial variables found.")
        return None, None, display_label

    var = st.sidebar.selectbox("Variable", vars_list, key=f"{key_prefix}_var")
    return ds, var, display_label


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="NetCDF Comparison Tool", layout="wide",
                       page_icon="🌊")
    st.markdown(OCEAN_CSS, unsafe_allow_html=True)

    # Header
    st.markdown("## 🌊 NetCDF Comparison Tool")
    st.markdown(
        '<p style="color:#4a9eca;font-size:0.85rem;margin-top:-10px;">'
        'Compare two NetCDF files — maps · statistics · depth profiles'
        '</p>',
        unsafe_allow_html=True,
    )
    st.divider()

    # --- Sidebar ---
    st.sidebar.markdown("## Files")
    ds_a, var_a, label_a = sidebar_file_panel("File A", "a")
    st.sidebar.divider()
    ds_b, var_b, label_b = sidebar_file_panel("File B", "b")

    st.sidebar.divider()
    st.sidebar.markdown('<div class="sidebar-section">Display Options</div>', unsafe_allow_html=True)
    cmap = st.sidebar.selectbox("Colormap", ["RdYlBu_r", "viridis", "turbo", "plasma", "coolwarm"])
    pct_lo, pct_hi = st.sidebar.slider("Percentile clip", 0, 50, (2, 98), step=1)

    manual_vmin = manual_vmax = manual_dmax = None

    manual_field = st.sidebar.checkbox("Manual field colorbar range")
    if manual_field:
        col1, col2 = st.sidebar.columns(2)
        manual_vmin = col1.number_input("vmin", value=0.0, format="%.4f", key="vmin")
        manual_vmax = col2.number_input("vmax", value=1.0, format="%.4f", key="vmax")

    manual_diff = st.sidebar.checkbox("Manual diff colorbar range")
    if manual_diff:
        manual_dmax = st.sidebar.number_input("Diff ±max", value=1.0, min_value=0.0001,
                                               format="%.4f", key="dmax")

    # --- Guard: both files must be loaded ---
    if ds_a is None or ds_b is None:
        st.info("Load two NetCDF files in the sidebar to begin.")
        return

    # --- Time index ---
    t_size_a = get_time_size(ds_a, var_a)
    t_size_b = get_time_size(ds_b, var_b)
    t_size = max(t_size_a, t_size_b)

    if t_size > 1:
        time_idx = st.sidebar.slider("Time index", 0, t_size - 1, 0)
    else:
        time_idx = 0

    # --- Extract arrays ---
    try:
        arr_a = extract_slice(ds_a, var_a, min(time_idx, max(t_size_a - 1, 0)))
        arr_b = extract_slice(ds_b, var_b, min(time_idx, max(t_size_b - 1, 0)))
    except Exception as e:
        st.error(f"Error extracting data: {e}")
        return

    depth_a = get_depth_coords(ds_a, var_a)
    depth_b = get_depth_coords(ds_b, var_b)
    is_3d_a = arr_a.ndim == 3
    is_3d_b = arr_b.ndim == 3
    is_3d = is_3d_a and is_3d_b

    # --- Shape mismatch check ---
    shape_a = arr_a.shape
    shape_b = arr_b.shape
    if shape_a != shape_b:
        st.error(
            f"Shape mismatch: File A `{var_a}` has shape {shape_a}, "
            f"File B `{var_b}` has shape {shape_b}. "
            "Select variables with matching dimensions."
        )
        return

    # --- Depth layer selector (for 3D) ---
    depth_idx = 0
    depth_values = None
    if is_3d:
        n_depths = arr_a.shape[0]
        depth_values = depth_a if depth_a is not None else (
            depth_b if depth_b is not None else np.arange(n_depths, dtype=float)
        )
        depth_labels = [f"Layer {i}: {depth_values[i]:.1f} m" for i in range(n_depths)]
        depth_idx = st.select_slider("Depth layer", options=list(range(n_depths)),
                                      format_func=lambda i: depth_labels[i])

    # --- Extract lat/lon coords for hover ---
    def _get_coord(ds, var, names):
        for name in names:
            if name in ds.coords:
                return ds.coords[name].values
        for dim in ds[var].dims:
            if dim.lower() in names:
                return ds.coords[dim].values if dim in ds.coords else None
        return None

    lat_names = {"lat", "latitude", "y", "nav_lat"}
    lon_names = {"lon", "longitude", "x", "nav_lon"}
    lats = _get_coord(ds_a, var_a, lat_names)
    lons = _get_coord(ds_a, var_a, lon_names)

    # --- 2D slices for current layer ---
    a2d = arr_a[depth_idx] if is_3d else arr_a
    b2d = arr_b[depth_idx] if is_3d else arr_b

    # --- Tabs ---
    tab1, tab2, tab3 = st.tabs(["  Maps  ", "  Statistics  ", "  Depth Profile  "])

    with tab1:
        render_maps(a2d, b2d, label_a, label_b, cmap, (pct_lo, pct_hi),
                    manual_vmin, manual_vmax, manual_dmax, lons=lons, lats=lats)
        st.markdown('<div class="section-label">Error Metrics — current layer</div>',
                    unsafe_allow_html=True)
        stats = compute_stats(a2d, b2d)
        render_stats_bar(stats)

    with tab2:
        if is_3d:
            df_stats = compute_all_depth_stats(arr_a, arr_b, depth_values)
            st.dataframe(
                df_stats.style.format({
                    "depth_m": "{:.1f}",
                    "rmse": "{:.4f}",
                    "mae": "{:.4f}",
                    "bias": "{:.4f}",
                    "pearson_r": "{:.4f}",
                    "n_valid": "{:,}",
                }),
                use_container_width=True,
            )
            csv = df_stats.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, "depth_stats.csv", "text/csv")
        else:
            stats = compute_stats(a2d, b2d)
            df_single = pd.DataFrame([stats])
            st.dataframe(df_single.style.format({
                "rmse": "{:.4f}", "mae": "{:.4f}",
                "bias": "{:.4f}", "pearson_r": "{:.4f}",
            }), use_container_width=True)

    with tab3:
        if is_3d:
            metric = st.radio("Metric", ["rmse", "mae"], horizontal=True,
                               format_func=str.upper)
            df_stats = compute_all_depth_stats(arr_a, arr_b, depth_values)
            render_depth_profile(df_stats, metric)
        else:
            st.info("Depth profile is only available for 3D (depth × lat × lon) variables.")


if __name__ == "__main__":
    main()
