import xarray as xr
import numpy as np
import plotly.graph_objects as go

# =========================
# 1. 读取数据
# =========================
ds = xr.open_dataset(
    # r"./output_data\recon\2025\recon_20250701.nc"
    r"./sit_2025_05.nc"
)

thetao = ds["thetao"].isel(time=0)
so = ds["so"].isel(time=0)

# =========================
# 2. 降采样（必须）
# =========================
thetao = thetao.isel(
    depth=slice(None, None, 3),
    latitude=slice(None, None, 6),
    longitude=slice(None, None, 6)
)
so = so.isel(
    depth=slice(None, None, 3),
    latitude=slice(None, None, 6),
    longitude=slice(None, None, 6)
)

# =========================
# 3. 构建统一坐标 (depth, lat, lon)
# =========================
depth = thetao.depth.values[:, None, None]
lat   = thetao.latitude.values[None, :, None]
lon   = thetao.longitude.values[None, None, :]

Depth = np.broadcast_to(depth, thetao.shape)
Lat   = np.broadcast_to(lat,   thetao.shape)
Lon   = np.broadcast_to(lon,   thetao.shape)

# =========================
# 4. 数据 & 掩码
# =========================
theta_vals = thetao.values
so_vals = so.values

mask = ~np.isnan(theta_vals) & ~np.isnan(so_vals)

# =========================
# 5. thetao 3D 文件
# =========================
theta_fig = go.Figure(
    go.Scatter3d(
        x=Lon[mask],
        y=Lat[mask],
        z=Depth[mask],   # 正深度
        mode="markers",
        marker=dict(
            size=2,
            color=theta_vals[mask],
            colorscale="Thermal",
            opacity=0.7,
            colorbar=dict(title="thetao (°C)")
        ),
        hovertemplate=
            "Lon: %{x:.2f}<br>"
            "Lat: %{y:.2f}<br>"
            "Depth: %{z:.1f} m<br>"
            "Thetao: %{marker.color:.2f} °C"
    )
)

theta_fig.update_layout(
    title="3D Scatter of thetao",
    scene=dict(
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        zaxis=dict(
            title="Depth (m)",
            autorange="reversed"
        )
    ),
    width=1100,
    height=850
)

theta_fig.write_html("thetao_3d.html")
print("✅ 已生成 thetao_3d.html")

# =========================
# 6. so 3D 文件
# =========================
so_fig = go.Figure(
    go.Scatter3d(
        x=Lon[mask],
        y=Lat[mask],
        z=Depth[mask],
        mode="markers",
        marker=dict(
            size=2,
            color=so_vals[mask],
            colorscale="Viridis",
            opacity=0.7,
            colorbar=dict(title="so (psu)")
        ),
        hovertemplate=
            "Lon: %{x:.2f}<br>"
            "Lat: %{y:.2f}<br>"
            "Depth: %{z:.1f} m<br>"
            "Salinity: %{marker.color:.2f} psu"
    )
)

so_fig.update_layout(
    title="3D Scatter of so",
    scene=dict(
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        zaxis=dict(
            title="Depth (m)",
            autorange="reversed"
        )
    ),
    width=1100,
    height=850
)

so_fig.write_html("so_3d.html")
print("✅ 已生成 so_3d.html")
