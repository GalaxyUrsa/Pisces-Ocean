/**
 * Ocean Sound Speed — frontend logic
 */

const state = {
  depthIdx:    0,
  mode:        "point",   // "point" | "transect"
  variable:    "ss",      // "ss" | "temp" | "salt"
  points:      [],
  meta:        null,
  colorRange:  null,      // [cmin, cmax] for 3D + 2D, or null
  depthRange:  null,
  valueRange:  null,
  varConfig:   { ss: null, temp: null, salt: null },
  drag: {
    active:        false,
    pointIdx:      null,   // which point (0 or 1) is being dragged
    hoverPointIdx: null,   // which point the mouse is hovering over
  },
};

const VAR_LABELS = { ss: "声速", temp: "温度", salt: "盐度" };
const VAR_UNITS  = { ss: "m/s",  temp: "°C",   salt: "PSU"  };

const VAR_DEFAULTS = {
  ss:   { min: 1480, max: 1560, colorscale: "Viridis"  },
  temp: { min: 0,    max: 35,   colorscale: "RdYlBu_r" },
  salt: { min: 30,   max: 40,   colorscale: "Blues"    },
};

const COLORSCALES = [
  "Viridis", "Plasma", "Inferno", "Magma", "Cividis",
  "RdYlBu_r", "RdBu_r", "Spectral_r",
  "Blues", "Greens", "YlOrRd",
  "Jet", "Turbo", "Rainbow",
];

const PLOTLY_CONFIG = {
  scrollZoom: true,
  displayModeBar: true,
  doubleClick: "reset",
  modeBarButtonsToRemove: ["select2d", "lasso2d"],
};
const PLOTLY_CONFIG_MAP = {
  scrollZoom: true,
  displayModeBar: true,
  doubleClick: "reset",
  plotGlPixelRatio: 1,
  modeBarButtonsToRemove: ["zoom2d", "select2d", "lasso2d", "zoomIn2d", "zoomOut2d", "autoScale2d"],
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function apiFetch(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API error ${res.status}: ${text}`);
  }
  return res.json();
}

function setClickInfo(text) {
  document.getElementById("map-click-info").textContent = text;
}

function setProfileTitle(text) {
  document.getElementById("profile-panel-title").textContent = text;
}

function setUploadStatus(msg, type = "") {
  const el = document.getElementById("upload-status");
  el.textContent = msg;
  el.className = "upload-status" + (type ? " " + type : "");
}

function _updateVarLabels(variable) {
  const label = VAR_LABELS[variable] || "值";
  const unit  = VAR_UNITS[variable]  || "";
  const labelEl = document.getElementById("value-range-label");
  const unitEl  = document.getElementById("value-range-unit");
  if (labelEl) labelEl.textContent = label;
  if (unitEl)  unitEl.textContent  = unit;
}

function _syncCbarInputs(range) {
  const minEl = document.getElementById("cmin");
  const maxEl = document.getElementById("cmax");
  if (!minEl || !maxEl) return;
  if (range) {
    minEl.value = range[0];
    maxEl.value = range[1];
  } else {
    minEl.value = "";
    maxEl.value = "";
  }
}

function _syncConfigUI(variable) {
  const cfg = state.varConfig[variable];
  if (!cfg) return;
  _syncCbarInputs([cfg.min, cfg.max]);

  const sel       = document.getElementById("colorscale-select");
  const minSwatch = document.getElementById("color-min-swatch");
  const minHex    = document.getElementById("color-min-hex");
  const maxSwatch = document.getElementById("color-max-swatch");
  const maxHex    = document.getElementById("color-max-hex");

  if (cfg.color_min && cfg.color_max) {
    if (sel) sel.value = "";
    if (minSwatch) minSwatch.value = cfg.color_min;
    if (minHex)    minHex.value    = cfg.color_min;
    if (maxSwatch) maxSwatch.value = cfg.color_max;
    if (maxHex)    maxHex.value    = cfg.color_max;
  } else {
    if (sel) sel.value = cfg.colorscale || "";
    if (minHex)    minHex.value    = "";
    if (maxHex)    maxHex.value    = "";
    if (minSwatch) minSwatch.value = "#000000";
    if (maxSwatch) maxSwatch.value = "#000000";
  }
}

// Drag helpers
let _profileDebounceTimer = null;
function _debouncedRenderProfile(delay = 150) {
  clearTimeout(_profileDebounceTimer);
  _profileDebounceTimer = setTimeout(() => renderProfile(), delay);
}

function _pixelToLatLon(gd, clientX, clientY) {
  const layout = gd._fullLayout;
  const rect   = gd.getBoundingClientRect();
  const px = clientX - rect.left - layout.margin.l;
  const py = clientY - rect.top  - layout.margin.t;
  const lon = layout.xaxis.p2d(px);
  const lat = layout.yaxis.p2d(py);
  const m = state.meta;
  return {
    lon: Math.max(m.lon_range[0], Math.min(m.lon_range[1], lon)),
    lat: Math.max(m.lat_range[0], Math.min(m.lat_range[1], lat)),
  };
}

function _updateMapMarkers() {
  const gd = document.getElementById("layer-map-graph");
  if (!gd || !state.points.length) return;
  Plotly.restyle(gd, {
    x: [state.points.map(p => p.lon)],
    y: [state.points.map(p => p.lat)],
  }, [1]);
  if (state.points.length === 2) {
    Plotly.restyle(gd, {
      x: [[state.points[0].lon, state.points[1].lon]],
      y: [[state.points[0].lat, state.points[1].lat]],
    }, [2]);
  }
}

// ---------------------------------------------------------------------------
// Upload screen
// ---------------------------------------------------------------------------

function initUploadScreen() {
  const input    = document.getElementById("nc-file-input");
  const dropZone = document.getElementById("drop-zone");

  input.addEventListener("change", () => {
    if (input.files[0]) handleFile(input.files[0]);
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  document.getElementById("reload-btn").addEventListener("click", () => {
    document.getElementById("main-screen").classList.add("hidden");
    document.getElementById("upload-screen").classList.remove("hidden");
    setUploadStatus("");
    input.value = "";
  });
}

async function handleFile(file) {
  if (!file.name.endsWith(".nc")) {
    setUploadStatus("请选择 .nc 格式的文件", "error");
    return;
  }

  setUploadStatus("正在上传并计算声速场，请稍候…", "loading");

  const form = new FormData();
  form.append("file", file);

  try {
    const result = await fetch("/api/upload", { method: "POST", body: form });
    if (!result.ok) {
      const err = await result.json();
      setUploadStatus("加载失败：" + (err.detail || result.statusText), "error");
      return;
    }
    const data = await result.json();
    setUploadStatus(`已加载：${data.filename}  (${data.shape.join(" × ")})`, "success");

    document.getElementById("upload-screen").classList.add("hidden");
    document.getElementById("main-screen").classList.remove("hidden");
    document.getElementById("loaded-filename").textContent = data.filename;

    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    await initMainScreen();
  } catch (e) {
    setUploadStatus("网络错误：" + e.message, "error");
  }
}

// ---------------------------------------------------------------------------
// Main screen
// ---------------------------------------------------------------------------

async function initMainScreen() {
  state.meta       = await apiFetch("/api/meta");
  state.points     = [];
  state.depthIdx   = 0;
  state.variable   = "ss";
  state.depthRange = null;
  state.valueRange = null;
  ["depth-min","depth-max","speed-min","speed-max"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.value = "";
  });
  // Load per-variable config from server (falls back to defaults if file missing)
  try {
    const saved = await apiFetch("/api/config");
    ["ss", "temp", "salt"].forEach(v => {
      state.varConfig[v] = saved[v] ? { ...VAR_DEFAULTS[v], ...saved[v] } : { ...VAR_DEFAULTS[v] };
    });
  } catch {
    ["ss", "temp", "salt"].forEach(v => { state.varConfig[v] = { ...VAR_DEFAULTS[v] }; });
  }
  const initCfg = state.varConfig["ss"];
  state.colorRange = [initCfg.min, initCfg.max];

  // Populate colorscale select
  const sel = document.getElementById("colorscale-select");
  if (sel) {
    sel.innerHTML = "";
    COLORSCALES.forEach(name => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    });
  }
  _syncConfigUI("ss");

  // Link color swatches ↔ hex inputs; handle preset/custom mutual exclusion
  function _linkColor(swatchId, hexId) {
    const swatch = document.getElementById(swatchId);
    const hex    = document.getElementById(hexId);
    if (!swatch || !hex) return;
    swatch.addEventListener("input", () => {
      hex.value = swatch.value;
      document.getElementById("colorscale-select").value = "";
    });
    hex.addEventListener("input", () => {
      if (/^#[0-9a-fA-F]{6}$/.test(hex.value)) {
        swatch.value = hex.value;
        document.getElementById("colorscale-select").value = "";
      }
    });
  }
  _linkColor("color-min-swatch", "color-min-hex");
  _linkColor("color-max-swatch", "color-max-hex");

  document.getElementById("colorscale-select").addEventListener("change", () => {
    document.getElementById("color-min-hex").value    = "";
    document.getElementById("color-max-hex").value    = "";
    document.getElementById("color-min-swatch").value = "#000000";
    document.getElementById("color-max-swatch").value = "#000000";
  });

  // Fill meta display
  const m = state.meta;
  const setText = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };
  setText("meta-lon",    `${m.lon_range[0].toFixed(1)}° – ${m.lon_range[1].toFixed(1)}°`);
  setText("meta-lat",    `${m.lat_range[0].toFixed(1)}° – ${m.lat_range[1].toFixed(1)}°`);
  setText("meta-depths", `${m.depths.length} 层`);
  setText("meta-grid",   `${m.grid_shape ? m.grid_shape.join(" × ") : "--"}`);
  setText("depth-index", `第 1 / ${m.depths.length} 层`);
  setText("depth-total", `共 ${m.depths.length} 层`);
  _updateVarLabels("ss");

  function _updateVolTitle(variable) {
    const el = document.getElementById("vol-panel-title");
    if (el) el.textContent = `3D ${VAR_LABELS[variable] || ""}场`;
  }
  _updateVolTitle("ss");

  // Variable cards
  document.querySelectorAll(".var-card").forEach(card => {
    card.classList.toggle("active", card.dataset.var === "ss");
    card.addEventListener("click", () => {
      document.querySelectorAll(".var-card").forEach(c => c.classList.remove("active"));
      card.classList.add("active");
      state.variable   = card.dataset.var;
      state.depthRange = null;
      state.valueRange = null;
      ["depth-min","depth-max","speed-min","speed-max"].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.value = "";
      });
      // Restore saved config for this variable
      const cfg = state.varConfig[state.variable];
      state.colorRange = cfg ? [cfg.min, cfg.max] : null;
      _syncConfigUI(state.variable);
      _updateVarLabels(state.variable);
      _updateVolTitle(state.variable);
      renderVolume();
      renderLayer(state.depthIdx, state.points);
      renderProfile();
    });
  });

  // Depth select + slider
  const depthSelect = document.getElementById("depth-select");
  const depthSlider = document.getElementById("depth-slider");
  depthSelect.innerHTML = "";
  state.meta.depths.forEach((d, i) => {
    const opt = document.createElement("option");
    opt.value       = i;
    opt.textContent = `${d.toFixed(1)} m`;
    depthSelect.appendChild(opt);
  });
  depthSlider.min   = 0;
  depthSlider.max   = state.meta.depths.length - 1;
  depthSlider.value = 0;

  depthSelect.addEventListener("change", () => {
    const i = parseInt(depthSelect.value);
    depthSlider.value = i;
    onDepthChange(i);
  });
  depthSlider.addEventListener("input", () => {
    const i = parseInt(depthSlider.value);
    depthSelect.value = i;
    onDepthChange(i);
  });

  // Mode radio
  document.querySelectorAll('input[name="mode"]').forEach((radio) => {
    radio.checked = radio.value === "point";
    radio.addEventListener("change", () => {
      state.mode   = radio.value;
      state.points = [];
      document.getElementById("mode-point").classList.toggle("active", state.mode === "point");
      document.getElementById("mode-transect").classList.toggle("active", state.mode === "transect");
      renderLayer(state.depthIdx, []);
      renderProfile();
    });
  });
  state.mode = "point";

  // Clear button
  document.getElementById("clear-btn").onclick = () => {
    state.points = [];
    renderLayer(state.depthIdx, []);
    renderProfile();
  };

  // Range controls
  document.getElementById("range-apply-btn").onclick = () => {
    const dMin = parseFloat(document.getElementById("depth-min").value);
    const dMax = parseFloat(document.getElementById("depth-max").value);
    const vMin = parseFloat(document.getElementById("speed-min").value);
    const vMax = parseFloat(document.getElementById("speed-max").value);
    state.depthRange = (!isNaN(dMin) && !isNaN(dMax)) ? [dMin, dMax] : null;
    state.valueRange = (!isNaN(vMin) && !isNaN(vMax)) ? [vMin, vMax] : null;
    renderProfile();
  };

  document.getElementById("range-reset-btn").onclick = () => {
    state.depthRange = null;
    state.valueRange = null;
    ["depth-min","depth-max","speed-min","speed-max"].forEach(id => {
      document.getElementById(id).value = "";
    });
    renderProfile();
  };
  // Colorbar range — unified control
  document.getElementById("cbar-apply").onclick = async () => {
    const cmin    = parseFloat(document.getElementById("cmin").value);
    const cmax    = parseFloat(document.getElementById("cmax").value);
    const cs      = document.getElementById("colorscale-select").value;
    const cMinHex = document.getElementById("color-min-hex").value.trim();
    const cMaxHex = document.getElementById("color-max-hex").value.trim();
    const hexRe   = /^#[0-9a-fA-F]{6}$/;
    const hasCustom = hexRe.test(cMinHex) && hexRe.test(cMaxHex);

    if (!isNaN(cmin) && !isNaN(cmax)) {
      state.varConfig[state.variable] = {
        min: cmin, max: cmax,
        colorscale:  hasCustom ? null : (cs || VAR_DEFAULTS[state.variable].colorscale),
        color_min:   hasCustom ? cMinHex : null,
        color_max:   hasCustom ? cMaxHex : null,
      };
      state.colorRange = [cmin, cmax];
      state.valueRange = [cmin, cmax];
      document.getElementById("speed-min").value = cmin;
      document.getElementById("speed-max").value = cmax;
      try { await apiFetch("/api/config", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(state.varConfig) }); } catch {}
      renderVolume();
      renderLayer(state.depthIdx, state.points);
      renderProfile();
    }
  };

  // Render charts
  await Promise.all([renderVolume(), renderLayer(0)]);

  const empty = emptyFigure("在右侧地图点击选点，查看垂直剖面或断面");
  Plotly.newPlot("profile-graph", empty.data, empty.layout, PLOTLY_CONFIG);
  setProfileTitle("剖面 / 断面");

  initMapDrag();
}

// ---------------------------------------------------------------------------
// Chart renderers
// ---------------------------------------------------------------------------

async function renderVolume() {
  const cfg = state.varConfig[state.variable] || {};
  const params = new URLSearchParams({ variable: state.variable });
  if (state.colorRange) {
    params.set("cmin", state.colorRange[0]);
    params.set("cmax", state.colorRange[1]);
  }
  if (cfg.color_min && cfg.color_max) {
    params.set("color_min", cfg.color_min);
    params.set("color_max", cfg.color_max);
  } else if (cfg.colorscale) {
    params.set("colorscale", cfg.colorscale);
  }
  const data = await apiFetch(`/api/volume?${params}`);
  Plotly.react("vol-graph", data.data, data.layout, PLOTLY_CONFIG);

  // Re-attach click handler (react may replace the element)
  document.getElementById("vol-graph").removeAllListeners &&
    document.getElementById("vol-graph").removeAllListeners("plotly_click");
  document.getElementById("vol-graph").on("plotly_click", (evt) => {
    if (!evt || !evt.points || !evt.points.length) return;
    const curveNumber = evt.points[0].curveNumber;
    if (curveNumber >= 0 && curveNumber < state.meta.depths.length) {
      document.getElementById("depth-select").value = curveNumber;
      document.getElementById("depth-slider").value = curveNumber;
      onDepthChange(curveNumber);
    }
  });
}

async function renderLayer(depthIdx, points = []) {
  const cfg = state.varConfig[state.variable] || {};
  const params = new URLSearchParams({ variable: state.variable });
  if (points.length) params.set("points", JSON.stringify(points));
  if (state.colorRange) {
    params.set("cmin", state.colorRange[0]);
    params.set("cmax", state.colorRange[1]);
  }
  if (cfg.color_min && cfg.color_max) {
    params.set("color_min", cfg.color_min);
    params.set("color_max", cfg.color_max);
  } else if (cfg.colorscale) {
    params.set("colorscale", cfg.colorscale);
  }
  const data = await apiFetch(`/api/layer/${depthIdx}?${params}`);

  // Preserve current zoom/pan before react() resets the layout
  const gd = document.getElementById("layer-map-graph");
  let savedRange = null;
  if (gd._fullLayout) {
    const xa = gd._fullLayout.xaxis;
    const ya = gd._fullLayout.yaxis;
    if (xa && ya && !xa.autorange && !ya.autorange) {
      savedRange = {
        "xaxis.range": [...xa.range],
        "yaxis.range": [...ya.range],
      };
    }
  }

  Plotly.react("layer-map-graph", data.figure.data, data.figure.layout, PLOTLY_CONFIG_MAP);

  if (savedRange) {
    Plotly.relayout("layer-map-graph", savedRange);
  }

  document.getElementById("layer-panel-title").textContent = data.title;

  attachMapClick();
}

async function renderProfile() {
  if (!state.points.length) {
    const empty = emptyFigure("在右侧地图点击选点，查看垂直剖面或断面");
    Plotly.react("profile-graph", empty.data, empty.layout, PLOTLY_CONFIG);
    setProfileTitle("剖面 / 断面");
    setClickInfo("");
    return;
  }

  if (state.mode === "point") {
    const p = state.points[state.points.length - 1];
    const data = await apiFetch("/api/profile", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        lat: p.lat, lon: p.lon,
        depth_idx:   state.depthIdx,
        variable:    state.variable,
        depth_range: state.depthRange,
        value_range: state.valueRange,
      }),
    });
    Plotly.react("profile-graph", data.figure.data, data.figure.layout, PLOTLY_CONFIG);
    setProfileTitle(data.title);
    setClickInfo(data.info);

  } else {
    if (state.points.length < 2) {
      const empty = emptyFigure("再点击一个点完成断面选取");
      Plotly.react("profile-graph", empty.data, empty.layout, PLOTLY_CONFIG);
      setProfileTitle("断面");
      const p1 = state.points[0];
      setClickInfo(`P1: ${p1.lat.toFixed(2)}°N, ${p1.lon.toFixed(2)}°E — 等待 P2`);
      return;
    }
    const [p1, p2] = state.points;
    const data = await apiFetch("/api/transect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        p1: { lat: p1.lat, lon: p1.lon },
        p2: { lat: p2.lat, lon: p2.lon },
        depth_idx:   state.depthIdx,
        variable:    state.variable,
        depth_range: state.depthRange,
        value_range: state.valueRange,
      }),
    });
    Plotly.react("profile-graph", data.figure.data, data.figure.layout, PLOTLY_CONFIG);
    setProfileTitle(data.title);
    setClickInfo(data.info);
  }
}

function emptyFigure(msg) {
  return {
    data: [],
    layout: {
      paper_bgcolor: "#161b22",
      plot_bgcolor:  "#0d1117",
      font:   { color: "#e6edf3" },
      margin: { l: 10, r: 10, t: 10, b: 10 },
      xaxis:  { visible: false },
      yaxis:  { visible: false },
      annotations: [{
        text: msg, x: 0.5, y: 0.5,
        xref: "paper", yref: "paper",
        showarrow: false,
        font: { size: 13, color: "#8b949e" },
      }],
    },
  };
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

function attachMapClick() {
  const el = document.getElementById("layer-map-graph");
  el.removeAllListeners && el.removeAllListeners("plotly_click");
  el.removeAllListeners && el.removeAllListeners("plotly_hover");
  el.removeAllListeners && el.removeAllListeners("plotly_unhover");

  el.on("plotly_hover", (evt) => {
    if (!evt || !evt.points || !evt.points.length) return;
    const pt = evt.points[0];
    if (pt.curveNumber === 1) {
      state.drag.hoverPointIdx = pt.pointIndex;
      el.style.cursor = "grab";
    } else {
      state.drag.hoverPointIdx = null;
      if (!state.drag.active) el.style.cursor = "";
    }
  });

  el.on("plotly_unhover", () => {
    if (!state.drag.active) {
      state.drag.hoverPointIdx = null;
      el.style.cursor = "";
    }
  });

  el.on("plotly_click", async (evt) => {
    if (state.drag.active) return;
    if (!evt || !evt.points || !evt.points.length) return;
    const pt = evt.points[0];
    if (pt.x == null || pt.y == null) return;
    if (pt.curveNumber >= 1) return;  // ignore clicks on marker/line traces

    const newPt = { lat: pt.y, lon: pt.x };
    if (state.mode === "point") {
      state.points = [newPt];
    } else {
      state.points = [...state.points, newPt].slice(-2);
    }

    await renderLayer(state.depthIdx, state.points);
    await renderProfile();
  });
}

function initMapDrag() {
  const el = document.getElementById("layer-map-graph");

  el.addEventListener("mousedown", (e) => {
    if (state.drag.hoverPointIdx === null) return;
    e.stopPropagation();
    e.preventDefault();
    state.drag.active   = true;
    state.drag.pointIdx = state.drag.hoverPointIdx;
    el.style.cursor = "grabbing";
    document.body.style.userSelect = "none";

    function onMove(e) {
      if (!state.drag.active) return;
      const { lat, lon } = _pixelToLatLon(el, e.clientX, e.clientY);
      state.points[state.drag.pointIdx] = { lat, lon };
      _updateMapMarkers();
      _debouncedRenderProfile(150);
    }

    function onUp() {
      state.drag.active = false;
      el.style.cursor = state.drag.hoverPointIdx !== null ? "grab" : "";
      document.body.style.userSelect = "";
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      clearTimeout(_profileDebounceTimer);
      renderLayer(state.depthIdx, state.points);
      renderProfile();
    }

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  }, { capture: true });
}

async function onDepthChange(depthIdx) {
  state.depthIdx = depthIdx;
  const total = state.meta.depths.length;
  const el = document.getElementById("depth-index");
  if (el) el.textContent = `第 ${depthIdx + 1} / ${total} 层`;
  await renderLayer(depthIdx, state.points);
  await renderProfile();
}

// ---------------------------------------------------------------------------
// Panel resizer
// ---------------------------------------------------------------------------

function initResizer() {
  const resizer = document.getElementById("resizer");
  const left    = document.getElementById("panel-3d");
  const right   = document.getElementById("panel-map");

  let startX, startLeftW, startRightW;

  resizer.addEventListener("mousedown", (e) => {
    e.preventDefault();
    startX      = e.clientX;
    startLeftW  = left.getBoundingClientRect().width;
    startRightW = right.getBoundingClientRect().width;
    resizer.classList.add("dragging");
    document.body.style.cursor     = "col-resize";
    document.body.style.userSelect = "none";

    const volEl     = document.getElementById("vol-graph");
    const mapEl     = document.getElementById("layer-map-graph");
    const sidebarEl = document.querySelector(".sidebar");
    volEl.style.pointerEvents     = "none";
    mapEl.style.pointerEvents     = "none";
    sidebarEl.style.pointerEvents = "none";

    let rafId = null;

    function onMove(e) {
      if (rafId) return;
      rafId = requestAnimationFrame(() => {
        rafId = null;
        const dx       = e.clientX - startX;
        const total    = startLeftW + startRightW;
        const newLeft  = Math.max(200, Math.min(total - 200, startLeftW + dx));
        const newRight = total - newLeft;
        left.style.flex  = "none";
        left.style.width = newLeft + "px";
        right.style.flex  = "none";
        right.style.width = newRight + "px";
      });
    }

    function onUp() {
      if (rafId) cancelAnimationFrame(rafId);
      resizer.classList.remove("dragging");
      document.body.style.cursor     = "";
      document.body.style.userSelect = "";
      volEl.style.pointerEvents     = "";
      mapEl.style.pointerEvents     = "";
      sidebarEl.style.pointerEvents = "";
      Plotly.Plots.resize(volEl);
      Plotly.Plots.resize(mapEl);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    }

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });
}

function initSidebarResizer() {
  const resizer    = document.getElementById("sidebar-resizer");
  const sidebar    = document.querySelector(".sidebar");
  const chartsCol  = document.querySelector(".charts-col");

  let startX, startW;

  resizer.addEventListener("mousedown", (e) => {
    e.preventDefault();
    startX = e.clientX;
    startW = sidebar.getBoundingClientRect().width;
    document.body.style.cursor     = "col-resize";
    document.body.style.userSelect = "none";
    sidebar.style.pointerEvents    = "none";

    let rafId = null;

    function onMove(e) {
      if (rafId) return;
      rafId = requestAnimationFrame(() => {
        rafId = null;
        const newW = Math.max(180, Math.min(480, startW + (e.clientX - startX)));
        sidebar.style.width = newW + "px";
      });
    }

    function onUp() {
      if (rafId) cancelAnimationFrame(rafId);
      document.body.style.cursor     = "";
      document.body.style.userSelect = "";
      sidebar.style.pointerEvents    = "";
      const volEl = document.getElementById("vol-graph");
      const mapEl = document.getElementById("layer-map-graph");
      if (volEl) Plotly.Plots.resize(volEl);
      if (mapEl) Plotly.Plots.resize(mapEl);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    }

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });
}

function initRowResizer() {
  const resizer  = document.getElementById("row-resizer");
  const topRow   = document.getElementById("main-row");
  const profPanel = document.querySelector(".panel-profile");

  let startY, startTopH, startBotH;

  resizer.addEventListener("mousedown", (e) => {
    e.preventDefault();
    startY     = e.clientY;
    startTopH  = topRow.getBoundingClientRect().height;
    startBotH  = profPanel.getBoundingClientRect().height;
    document.body.style.cursor     = "row-resize";
    document.body.style.userSelect = "none";

    const volEl  = document.getElementById("vol-graph");
    const mapEl  = document.getElementById("layer-map-graph");
    const profEl = document.getElementById("profile-graph");
    volEl.style.pointerEvents  = "none";
    mapEl.style.pointerEvents  = "none";
    profEl.style.pointerEvents = "none";

    let rafId = null;

    function onMove(e) {
      if (rafId) return;
      rafId = requestAnimationFrame(() => {
        rafId = null;
        const dy      = e.clientY - startY;
        const newTop  = Math.max(150, startTopH + dy);
        const newBot  = Math.max(120, startBotH - dy);
        topRow.style.flex   = "none";
        topRow.style.height = newTop + "px";
        profPanel.style.flex   = "none";
        profPanel.style.height = newBot + "px";
      });
    }

    function onUp() {
      if (rafId) cancelAnimationFrame(rafId);
      document.body.style.cursor     = "";
      document.body.style.userSelect = "";
      volEl.style.pointerEvents  = "";
      mapEl.style.pointerEvents  = "";
      profEl.style.pointerEvents = "";
      Plotly.Plots.resize(volEl);
      Plotly.Plots.resize(mapEl);
      Plotly.Plots.resize(profEl);
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
    }

    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
  initUploadScreen();
  initResizer();
  initSidebarResizer();
  initRowResizer();

  fetch("/api/status").then(r => r.json()).then(async s => {
    if (s.ready) {
      document.getElementById("upload-screen").classList.add("hidden");
      document.getElementById("main-screen").classList.remove("hidden");
      document.getElementById("loaded-filename").textContent = "（命令行预加载）";
      await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
      initMainScreen();
    }
  });
});
