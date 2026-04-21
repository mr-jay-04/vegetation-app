import streamlit as st
import numpy as np
import rasterio
from rasterio.io import MemoryFile
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import io
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Vegetation Analyser",
    page_icon="🌿",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .stAlert { border-radius: 8px; }
    div[data-testid="metric-container"] {
        background: #f8f9fa;
        border: 0.5px solid #e0e0e0;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .section-title {
        font-size: 11px;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 4px;
    }
    .chip-available {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        background: #eafaf1;
        color: #1e8449;
        margin: 2px;
    }
    .chip-unavailable {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 12px;
        background: #f4f4f4;
        color: #aaa;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
INDEX_BANDS = {
    "NDVI": ["red", "nir"],
    "NDWI": ["green", "nir"],
    "SAVI": ["red", "nir"],
    "VARI": ["red", "green", "blue"],
}

INDEX_FORMULAS = {
    "NDVI": "(NIR − Red) / (NIR + Red)",
    "NDWI": "(Green − NIR) / (Green + NIR)",
    "SAVI": "((NIR − Red) / (NIR + Red + 0.5)) × 1.5",
    "VARI": "(Green − Red) / (Green + Red − Blue)",
}

INDEX_DESC = {
    "NDVI": "Overall vegetation health",
    "NDWI": "Water content in vegetation",
    "SAVI": "Vegetation with soil correction",
    "VARI": "Visible greenness (RGB only)",
}

FUSION_REQUIRES = {
    "NDVI + NDWI — Water stress":       ("NDVI", "NDWI"),
    "NDVI + SAVI — Soil interference":  ("NDVI", "SAVI"),
    "NDVI + VARI — Hidden stress":      ("NDVI", "VARI"),
}

HEALTH_THRESHOLDS = [
    (-1.0,  -0.1, "Water / Non-veg",   "#cfe2f3", "#1a5276"),
    (-0.1,   0.1, "Bare soil",         "#fdebd0", "#784212"),
    ( 0.1,   0.2, "Very sparse",       "#fadbd8", "#922b21"),
    ( 0.2,  0.35, "Sparse / Stressed", "#fef9e7", "#7d6608"),
    (0.35,   0.5, "Moderate",          "#eafaf1", "#1e8449"),
    ( 0.5,  0.65, "Healthy",           "#d5f5e3", "#1a5e34"),
    ( 0.65,  1.0, "Very Healthy",      "#a9dfbf", "#0b3d25"),
]

WATER_STRESS_ZONES = {
    0: ("Bare / dry",         "#FFFF00"), # Yellow
    1: ("Waterlogged/Soil",   "#00FFFF"), # Cyan
    2: ("dense canopy",       "#FF0000"), # Red
    3: ("Healthy",            "#00FF00"), # Green
}

HIDDEN_STRESS_ZONES = {
    0: ("No anomaly",        "#B2BABB"),
    1: ("Hidden stress",     "#E74C3C"),
    2: ("Confirmed healthy", "#27AE60"),
}

INDEX_LEGENDS = {
    "NDVI": [
        ("<= -0.6", "#D7191C", "#FFF"),
        ("-0.6 to -0.2", "#FDAE61", "#000"),
        ("-0.2 to 0.2", "#FFFFBF", "#000"),
        ("0.2 to 0.6", "#A6D96A", "#000"),
        ("> 0.6", "#1A9641", "#FFF")
    ],
    "SAVI": [
        ("<= 0.2", "#edf8e9", "#000"),
        ("0.2 to 0.4", "#bae4b3", "#000"),
        ("0.4 to 0.6", "#74c476", "#000"),
        ("0.6 to 0.8", "#31a354", "#FFF"),
        ("> 0.8", "#006d2c", "#FFF")
    ],
    "NDWI": [
        ("Dry/Veg", "#8B5E3C", "#FFF"),
        ("Moist", "#f4f4f4", "#000"),
        ("Water", "#2980B9", "#FFF")
    ],
    "VARI": [
        ("<= -0.25", "#d73027", "#FFF"),
        ("-0.25 to 0.0", "#fdae61", "#000"),
        ("0.0 to 0.25", "#a6d96a", "#000"),
        ("> 0.25", "#1a9850", "#FFF")
    ]
}

# ── SESSION STATE INIT ────────────────────────────────────────────────────────
_state_defaults = {
    "bands":            {},
    "profile_ref":      None,
    "computed_indices": {},
    "cluster_results":  {},
    "upload_mode":      None,
    "band_fingerprint": None,
}
for k, v in _state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CORE HELPERS ──────────────────────────────────────────────────────────────
def read_band_bytes(file_bytes: bytes, band_num: int = 1):
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            data    = src.read(band_num).astype(np.float32)
            profile = src.profile.copy()
            meta    = {
                "crs": str(src.crs), "transform": src.transform,
                "width": src.width,  "height": src.height,
                "count": src.count,
            }
    return data, profile, meta


def read_band_from_file(uploaded_file, band_num: int = 1):
    fb = uploaded_file.read()
    uploaded_file.seek(0)
    return read_band_bytes(fb, band_num)


def safe_divide(num, den, eps=1e-10):
    den_s  = np.where(np.abs(den) < eps, eps, den)
    result = num / den_s
    return np.where(np.abs(den) < eps, np.nan, result)


@st.cache_data(show_spinner=False)
def _compute_index_cached(name, r=None, nir=None, g=None, b=None):
    if name == "NDVI": return safe_divide(nir - r,   nir + r)
    if name == "NDWI": return safe_divide(g   - nir, g   + nir)
    if name == "SAVI": return safe_divide(nir - r,   nir + r + 0.5) * 1.5
    if name == "VARI": return safe_divide(g   - r,   g   + r - b)


def compute_and_store(name: str):
    bd  = st.session_state.bands
    arr = _compute_index_cached(
        name,
        r=bd.get("red"),
        nir=bd.get("nir"),
        g=bd.get("green"),
        b=bd.get("blue"),
    )
    st.session_state.computed_indices[name] = arr
    return arr


@st.cache_data(show_spinner=False)
def _run_kmeans_cached(index_array, k, sample_size=50_000):
    flat  = index_array.flatten()
    mask  = np.isfinite(flat)
    valid = flat[mask].reshape(-1, 1)
    km    = KMeans(n_clusters=k, random_state=42, n_init=3)
    if len(valid) > sample_size:
        rng  = np.random.default_rng(42)
        sidx = rng.choice(len(valid), size=sample_size, replace=False)
        km.fit(valid[sidx])
        labels = km.predict(valid)
    else:
        labels = km.fit_predict(valid)
    result       = np.full(flat.shape, -9999, dtype=np.int32)
    result[mask] = labels
    return result.reshape(index_array.shape)


MAX_DISPLAY_PX = 500

def downsample(array, max_dim=MAX_DISPLAY_PX):
    h, w = array.shape
    if max(h, w) <= max_dim:
        return array
    scale   = max_dim / max(h, w)
    new_h   = max(1, int(h * scale))
    new_w   = max(1, int(w * scale))
    row_idx = np.linspace(0, h - 1, new_h, dtype=int)
    col_idx = np.linspace(0, w - 1, new_w, dtype=int)
    return array[np.ix_(row_idx, col_idx)]


def vectorized_health_labels(arr):
    labels = np.full(arr.shape, "Very Healthy", dtype=object)
    for lo, hi, label, _, _ in HEALTH_THRESHOLDS:
        labels[(arr >= lo) & (arr < hi)] = label
    return labels


def available_indices(bands: dict) -> dict:
    return {
        name: all(b in bands for b in needed)
        for name, needed in INDEX_BANDS.items()
    }


def zone_stats_html(zone_array, zone_dict, nodata_val=-1):
    total = int(np.sum(zone_array != nodata_val))
    cards = ""
    for zid, info in zone_dict.items():
        color = info[1]; label = info[0]
        count = int(np.sum(zone_array == zid))
        pct   = count / total * 100 if total > 0 else 0
        cards += (
            f'<div style="flex:1;min-width:130px;background:{color}22;border:0.5px solid {color};'
            f'border-radius:8px;padding:10px;text-align:center;">'
            f'<div style="font-size:17px;font-weight:600;color:{color};">{pct:.1f}%</div>'
            f'<div style="font-size:12px;color:#444;margin-top:2px;">{label}</div>'
            f'<div style="font-size:10px;color:#888;">{count:,} px</div></div>'
        )
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;">{cards}</div>'


# ── PLOTLY FIGURES ────────────────────────────────────────────────────────────
def _base_layout(height=480):
    return dict(
        margin=dict(l=0, r=0, t=0, b=0), height=height,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed", zeroline=False),
        hoverlabel=dict(bgcolor="white", bordercolor="#ccc", font_size=13),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )


@st.cache_data(show_spinner=False)
def make_index_fig(index_array, index_name):
    ds = downsample(index_array)
    
    customdata = None
    hovertemplate = f"<b>{index_name}: %{{z:.4f}}</b><br>Pixel: (%{{x}}, %{{y}})<extra></extra>"
    
    if index_name == "NDVI":
        cscale = [
            [0.0, "#D7191C"], [0.2, "#D7191C"],
            [0.2, "#FDAE61"], [0.4, "#FDAE61"],
            [0.4, "#FFFFBF"], [0.6, "#FFFFBF"],
            [0.6, "#A6D96A"], [0.8, "#A6D96A"],
            [0.8, "#1A9641"], [1.0, "#1A9641"]
        ]
        zmin, zmax = -1.0, 1.0
        tickvals = [-0.8, -0.4, 0.0, 0.4, 0.8]
        ticktext = ["<= -0.6", "-0.6 to -0.2", "-0.2 to 0.2", "0.2 to 0.6", "> 0.6"]
        
    elif index_name == "SAVI":
        cscale = [
            [0.0, "#edf8e9"], [0.2, "#edf8e9"],
            [0.2, "#bae4b3"], [0.4, "#bae4b3"],
            [0.4, "#74c476"], [0.6, "#74c476"],
            [0.6, "#31a354"], [0.8, "#31a354"],
            [0.8, "#006d2c"], [1.0, "#006d2c"]
        ]
        zmin, zmax = 0.0, 1.0 
        tickvals = [0.1, 0.3, 0.5, 0.7, 0.9]
        ticktext = ["<= 0.2", "0.2 to 0.4", "0.4 to 0.6", "0.6 to 0.8", "> 0.8"]

    elif index_name == "NDWI":
        cscale = [
            [0.00, "#8B5E3C"], [0.33, "#8B5E3C"],
            [0.33, "#f4f4f4"], [0.66, "#f4f4f4"],
            [0.66, "#2980B9"], [1.00, "#2980B9"]
        ]
        zmin, zmax = -1.0, 1.0
        tickvals = [-0.66, 0.0, 0.66]
        ticktext = ["Dry/Veg (< -0.3)", "Moist (-0.3 to 0.3)", "Water (> 0.3)"]
        
    elif index_name == "VARI":
        cscale = [
            [0.00, "#d73027"], [0.25, "#d73027"],
            [0.25, "#fdae61"], [0.50, "#fdae61"],
            [0.50, "#a6d96a"], [0.75, "#a6d96a"],
            [0.75, "#1a9850"], [1.00, "#1a9850"]
        ]
        zmin, zmax = -0.5, 0.5
        tickvals = [-0.375, -0.125, 0.125, 0.375]
        ticktext = ["<= -0.25", "-0.25 to 0.0", "0.0 to 0.25", "> 0.25"]

    else:
        cscale = "RdYlGn"
        zmin, zmax = -0.5, 0.5
        tickvals, ticktext = None, None

    cbar_dict = dict(title=dict(text=index_name, side="right"), thickness=14, len=0.9)
    if tickvals and ticktext:
        # Added tickmode="array" to force Plotly to use the custom labels
        cbar_dict.update(tickmode="array", tickvals=tickvals, ticktext=ticktext)

    heatmap_kwargs = dict(
        z=ds, colorscale=cscale, zmin=zmin, zmax=zmax,
        colorbar=cbar_dict
    )
    
    if customdata is not None:
        heatmap_kwargs["customdata"] = customdata
        
    fig = go.Figure(go.Heatmap(**heatmap_kwargs, hovertemplate=hovertemplate))
    fig.update_layout(**_base_layout())
    return fig

@st.cache_data(show_spinner=False)
def make_cluster_fig(cluster_array, k):
    ds = downsample(cluster_array).astype(float)
    ds[ds == -9999] = np.nan
    colors = ["#1565c0","#2e7d32","#ef6c00","#6a1b9a","#c62828","#00695c","#f9a825","#4e342e"]
    cscale = [[i/(k-1) if k > 1 else 0, colors[i % len(colors)]] for i in range(k)]
    fig = go.Figure(go.Heatmap(
        z=ds, colorscale=cscale, zmin=0, zmax=k-1,
        hovertemplate="<b>Cluster: %{z:.0f}</b><br>Pixel: (%{x}, %{y})<extra></extra>",
        colorbar=dict(title=dict(text="Cluster", side="right"), thickness=14,
                      tickvals=list(range(k)), ticktext=[f"Cluster {i}" for i in range(k)], len=0.9),
    ))
    fig.update_layout(**_base_layout())
    return fig


@st.cache_data(show_spinner=False)
def make_water_stress_fig(zone_array, ndvi_ds, ndwi_ds):
    ds      = downsample(zone_array.astype(float))
    ds[ds == -1] = np.nan
    ds_ndvi = downsample(ndvi_ds)
    ds_ndwi = downsample(ndwi_ds)
    
    zlabels = np.full(ds.shape, "NoData", dtype=object)
    for zid, (label, _) in WATER_STRESS_ZONES.items():
        zlabels[ds == zid] = label
        
    # Safely mapped blocks to trap the integers 0, 1, 2, 3 without bleeding
    cscale = [
        [0.00, WATER_STRESS_ZONES[0][1]], [0.25, WATER_STRESS_ZONES[0][1]],
        [0.25, WATER_STRESS_ZONES[1][1]], [0.50, WATER_STRESS_ZONES[1][1]],
        [0.50, WATER_STRESS_ZONES[2][1]], [0.75, WATER_STRESS_ZONES[2][1]],
        [0.75, WATER_STRESS_ZONES[3][1]], [1.00, WATER_STRESS_ZONES[3][1]],
    ]
    
    custom = np.stack([zlabels, ds_ndvi, ds_ndwi], axis=-1)
    
    fig = go.Figure(go.Heatmap(
        z=ds, colorscale=cscale, zmin=0, zmax=3, customdata=custom,
        hovertemplate="<b>%{customdata[0]}</b><br>NDVI: %{customdata[1]:.4f}<br>NDWI: %{customdata[2]:.4f}<br>Pixel: (%{x}, %{y})<extra></extra>",
        colorbar=dict(
            title=dict(text="Zone", side="right"), thickness=14, len=0.9,
            tickmode="array", # Forces custom text
            tickvals=[0, 1, 2, 3], 
            ticktext=[WATER_STRESS_ZONES[i][0] for i in range(4)]
        ),
    ))
    fig.update_layout(**_base_layout())
    return fig


@st.cache_data(show_spinner=False)
def make_soil_interference_fig(ndvi_array, diff_array, unreliable_mask):
    ds_ndvi       = downsample(ndvi_array)
    ds_diff       = downsample(diff_array)
    ds_unreliable = downsample(unreliable_mask.astype(np.float32))
    
    # Use the 5-band QGIS colorscale for the base NDVI layer
    cscale = [
        [0.0, "#D7191C"], [0.2, "#D7191C"],
        [0.2, "#FDAE61"], [0.4, "#FDAE61"],
        [0.4, "#FFFFBF"], [0.6, "#FFFFBF"],
        [0.6, "#A6D96A"], [0.8, "#A6D96A"],
        [0.8, "#1A9641"], [1.0, "#1A9641"]
    ]
    
    reliability = np.where(ds_unreliable > 0.5, "Unreliable (soil interference)", "Reliable")
    custom = np.stack([reliability, ds_diff], axis=-1)
    
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=ds_ndvi, colorscale=cscale, zmin=-1.0, zmax=1.0, customdata=custom,
        hovertemplate="<b>NDVI: %{z:.4f}</b><br>Reliability: %{customdata[0]}<br>|NDVI−SAVI|: %{customdata[1]:.4f}<br>Pixel: (%{x}, %{y})<extra></extra>",
        colorbar=dict(
            title=dict(text="NDVI", side="right"), thickness=14, len=0.9, x=1.02,
            tickmode="array",
            tickvals=[-0.8, -0.4, 0.0, 0.4, 0.8],
            ticktext=["<= -0.6", "-0.6 to -0.2", "-0.2 to 0.2", "0.2 to 0.6", "> 0.6"]
        ),
    ))
    
    try:
        import base64
        from PIL import Image as PILImage
        h, w    = ds_unreliable.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        mask    = ds_unreliable > 0.5
        overlay[mask]  = [220, 53, 69, 140]
        overlay[~mask] = [0,   0,  0,   0]
        pil_img = PILImage.fromarray(overlay, mode="RGBA")
        buf     = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64     = base64.b64encode(buf.getvalue()).decode()
        fig.add_layout_image(dict(
            source=f"data:image/png;base64,{b64}",
            xref="x", yref="y", x=0, y=0,
            sizex=w, sizey=h, sizing="stretch", opacity=1.0, layer="above",
        ))
        fig.update_xaxes(range=[0, w])
        fig.update_yaxes(range=[h, 0])
    except ImportError:
        pass
        
    fig.update_layout(**_base_layout())
    return fig

@st.cache_data(show_spinner=False)
def make_hidden_stress_fig(zone_array, ndvi_ds, vari_ds):
    ds      = downsample(zone_array.astype(float))
    ds[ds == -1] = np.nan
    ds_ndvi = downsample(ndvi_ds)
    ds_vari = downsample(vari_ds)
    zlabels = np.full(ds.shape, "NoData", dtype=object)
    for zid, (label, _) in HIDDEN_STRESS_ZONES.items():
        zlabels[ds == zid] = label
    cscale = [
        [0.00, HIDDEN_STRESS_ZONES[0][1]], [0.33, HIDDEN_STRESS_ZONES[0][1]],
        [0.33, HIDDEN_STRESS_ZONES[1][1]], [0.66, HIDDEN_STRESS_ZONES[1][1]],
        [0.66, HIDDEN_STRESS_ZONES[2][1]], [1.00, HIDDEN_STRESS_ZONES[2][1]],
    ]
    custom = np.stack([zlabels, ds_ndvi, ds_vari], axis=-1)
    fig = go.Figure(go.Heatmap(
        z=ds, colorscale=cscale, zmin=0, zmax=2, customdata=custom,
        hovertemplate="<b>%{customdata[0]}</b><br>NDVI: %{customdata[1]:.4f}<br>VARI: %{customdata[2]:.4f}<br>Pixel: (%{x}, %{y})<extra></extra>",
        colorbar=dict(title=dict(text="Zone", side="right"), thickness=14,
                      tickvals=[0,1,2], ticktext=[HIDDEN_STRESS_ZONES[i][0] for i in range(3)], len=0.9),
    ))
    fig.update_layout(**_base_layout())
    return fig


# ── FUSION COMPUTATIONS ───────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def compute_water_stress_zones(ndvi_array, ndwi_array, ndvi_thresh=0.3, ndwi_thresh=0.0):
    valid   = np.isfinite(ndvi_array) & np.isfinite(ndwi_array)
    zones   = np.full(ndvi_array.shape, -1, dtype=np.int8)
    hi_ndvi = ndvi_array >= ndvi_thresh
    hi_ndwi = ndwi_array >= ndwi_thresh
    zones[valid & ~hi_ndvi & ~hi_ndwi] = 0
    zones[valid & ~hi_ndvi &  hi_ndwi] = 1
    zones[valid &  hi_ndvi & ~hi_ndwi] = 2
    zones[valid &  hi_ndvi &  hi_ndwi] = 3
    return zones


@st.cache_data(show_spinner=False)
def compute_soil_interference(ndvi_array, savi_array, percentile=75):
    diff       = np.abs(ndvi_array - savi_array)
    valid_diff = diff[np.isfinite(diff)]
    threshold  = float(np.nanpercentile(valid_diff, percentile)) if len(valid_diff) > 0 else 0.1
    unreliable = (diff >= threshold) & np.isfinite(diff)
    return diff, unreliable, threshold


@st.cache_data(show_spinner=False)
def compute_hidden_stress(ndvi_array, vari_array, vari_thresh=0.1, ndvi_thresh=0.3):
    valid   = np.isfinite(ndvi_array) & np.isfinite(vari_array)
    zones   = np.full(ndvi_array.shape, -1, dtype=np.int8)
    hi_vari = vari_array >= vari_thresh
    hi_ndvi = ndvi_array >= ndvi_thresh
    zones[valid & ~hi_vari & ~hi_ndvi] = 0
    zones[valid &  hi_vari & ~hi_ndvi] = 1
    zones[valid &  hi_vari &  hi_ndvi] = 2
    zones[valid & ~hi_vari &  hi_ndvi] = 0
    return zones


# ── EXPORT ────────────────────────────────────────────────────────────────────
def array_to_geotiff_bytes(array, profile):
    profile = profile.copy()
    profile.update(dtype=rasterio.float32, count=1, nodata=-9999,
                   compress="lzw", driver="GTiff")
    for key in ["blockxsize", "blockysize", "tiled"]:
        profile.pop(key, None)
    buf = io.BytesIO()
    with rasterio.open(buf, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(array), array, -9999).astype(np.float32), 1)
    buf.seek(0)
    return buf.read()


def cluster_to_geotiff_bytes(array, profile):
    profile = profile.copy()
    profile.update(dtype=rasterio.int32, count=1, nodata=-9999,
                   compress="lzw", driver="GTiff")
    for key in ["blockxsize", "blockysize", "tiled"]:
        profile.pop(key, None)
    buf = io.BytesIO()
    with rasterio.open(buf, "w", **profile) as dst:
        dst.write(array.astype(np.int32), 1)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Vegetation Analyser")
    st.markdown("---")
    st.markdown('<div class="section-title">Upload mode</div>', unsafe_allow_html=True)
    upload_mode = st.radio(
        "mode", ["Individual bands", "RGB image (VARI only)"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<div class="section-title">K-Means clusters</div>', unsafe_allow_html=True)
    k_val = st.slider("K", min_value=2, max_value=8, value=3, step=1,
                      label_visibility="collapsed")
    st.caption(f"K = {k_val} clusters")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — UPLOAD
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("## Step 1 — Upload imagery")

bands_loaded = {}
profile_ref  = None

if upload_mode == "Individual bands":
    st.caption("Upload any subset of bands. Only indices computable from your uploads will be shown.")
    uc = st.columns(4)
    band_specs = [
        ("red",   "Red band",   "#E74C3C"),
        ("green", "Green band", "#27AE60"),
        ("blue",  "Blue band",  "#2980B9"),
        ("nir",   "NIR band",   "#8E44AD"),
    ]
    for i, (bname, blabel, bcolor) in enumerate(band_specs):
        with uc[i]:
            st.markdown(
                f'<div style="font-size:11px;font-weight:600;color:{bcolor};">{blabel}</div>',
                unsafe_allow_html=True
            )
            f = st.file_uploader(blabel, type=["tif", "tiff"], key=f"ub_{bname}",
                                 label_visibility="collapsed")
            if f:
                arr, prof, meta = read_band_from_file(f, 1)
                bands_loaded[bname] = arr
                if profile_ref is None:
                    profile_ref = prof
                st.caption(f"{meta['width']}×{meta['height']} px")

    shapes = list({v.shape for v in bands_loaded.values()})
    if len(shapes) > 1:
        st.error(f"Band dimensions don't match: {shapes}. All bands must be the same size.")
        bands_loaded = {}

elif upload_mode == "RGB image (VARI only)":
    st.caption("RGB combined image — only VARI can be computed (no NIR channel).")
    rgb_f = st.file_uploader("Upload RGB image", type=["tif","tiff","jpg","jpeg","png"],
                              key="ub_rgb")
    if rgb_f:
        r_arr, prof, meta = read_band_from_file(rgb_f, 1)
        g_arr, _, _       = read_band_from_file(rgb_f, 2)
        b_arr, _, _       = read_band_from_file(rgb_f, 3)
        bands_loaded = {"red": r_arr, "green": g_arr, "blue": b_arr}
        profile_ref  = prof
        st.info(f"RGB loaded — {meta['width']}×{meta['height']} px | R=1, G=2, B=3")

# ── Persist bands ─────────────────────────────────────────────────────────────
if bands_loaded:
    new_band_keys = set(bands_loaded.keys())
    old_band_keys = set(st.session_state.bands.keys())
    if new_band_keys != old_band_keys:
        st.session_state.computed_indices = {}
        st.session_state.cluster_results  = {}
    st.session_state.bands       = bands_loaded
    st.session_state.profile_ref = profile_ref

bands       = st.session_state.bands
profile_ref = st.session_state.get("profile_ref")


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — SELECT & COMPUTE INDICES
# ─────────────────────────────────────────────────────────────────────────────
if not bands:
    st.info("Upload at least one band to continue.")
    st.stop()

st.markdown("---")
st.markdown("## Step 2 — Select indices to compute")

avail = available_indices(bands)

chip_html = ""
for idx_name, can_compute in avail.items():
    needed  = INDEX_BANDS[idx_name]
    missing = [b for b in needed if b not in bands]
    if can_compute:
        chip_html += f'<span class="chip-available">✔ {idx_name}</span>'
    else:
        chip_html += f'<span class="chip-unavailable">✖ {idx_name} (needs {", ".join(missing)})</span>'
st.markdown(chip_html, unsafe_allow_html=True)
st.markdown("")

computable = [name for name, ok in avail.items() if ok]

if not computable:
    st.warning("No indices can be computed from the uploaded bands. Upload more bands.")
    st.stop()

selected_indices = st.multiselect(
    "Select indices to compute",
    options=computable,
    default=[computable[0]],
    format_func=lambda x: f"{x} — {INDEX_DESC[x]}",
)

for idx in selected_indices:
    st.caption(f"`{idx}`: {INDEX_FORMULAS[idx]}")

if not selected_indices:
    st.warning("Select at least one index.")
    st.stop()

if st.button("Compute selected indices", type="primary"):
    prog  = st.progress(0, text="Starting...")
    total = len(selected_indices)

    for i, idx_name in enumerate(selected_indices):
        prog.progress(int((i / total) * 80), text=f"Computing {idx_name} ({i+1}/{total})...")
        bd  = st.session_state.bands
        arr = _compute_index_cached(
            idx_name,
            r=bd.get("red"),
            nir=bd.get("nir"),
            g=bd.get("green"),
            b=bd.get("blue"),
        )
        st.session_state.computed_indices[idx_name] = arr

    prog.progress(85, text=f"Running K-Means (K={k_val})...")
    for idx_name in selected_indices:
        arr = st.session_state.computed_indices[idx_name]
        st.session_state.cluster_results[idx_name] = _run_kmeans_cached(arr, k_val)

    prog.progress(100, text="Done!")
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 RESULTS
# ─────────────────────────────────────────────────────────────────────────────
computed = st.session_state.computed_indices

if computed:
    st.markdown("---")
    st.markdown("### Index results")

    for idx_name, idx_arr in computed.items():
        valid_vals = idx_arr[np.isfinite(idx_arr)]
        with st.expander(f"{idx_name} — {INDEX_DESC[idx_name]}", expanded=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Max",      f"{np.nanmax(valid_vals):.4f}")
            m2.metric("Mean",     f"{np.nanmean(valid_vals):.4f}")
            m3.metric("Min",      f"{np.nanmin(valid_vals):.4f}")
            m4.metric("Valid px", f"{len(valid_vals):,}")

            t1, t2 = st.tabs([f"{idx_name} Map", "Cluster Map"])
            with t1:
                st.plotly_chart(
                    make_index_fig(idx_arr, idx_name),
                    width="stretch",
                    key=f"index_fig_{idx_name}",
                )
                
                # Automatically render the HTML colored boxes below the map for EVERY index
                if idx_name in INDEX_LEGENDS:
                    st.markdown(f"##### {idx_name} scale")
                    legend_data = INDEX_LEGENDS[idx_name]
                    hcols = st.columns(len(legend_data))
                    for hi, (label, bg, tc) in enumerate(legend_data):
                        with hcols[hi]:
                            st.markdown(
                                f'<div style="background:{bg};color:{tc};padding:6px;border-radius:6px;'
                                f'font-size:11px;text-align:center;font-weight:600;border:1px solid #ccc;">'
                                f'{label}</div>',
                                unsafe_allow_html=True
                            )
            with t2:
                cl_arr = st.session_state.cluster_results.get(idx_name)
                if cl_arr is not None:
                    # ✅ unique key per index
                    st.plotly_chart(
                        make_cluster_fig(cl_arr, k_val),
                        width="stretch",
                        key=f"cluster_fig_{idx_name}",
                    )
                    st.markdown("##### Cluster distribution")
                    dcols  = st.columns(k_val)
                    colors = ["#1565c0","#2e7d32","#ef6c00","#6a1b9a","#c62828","#00695c","#f9a825","#4e342e"]
                    total_v = int(np.sum(cl_arr != -9999))
                    for ki in range(k_val):
                        count = int(np.sum(cl_arr == ki))
                        pct   = count / total_v * 100 if total_v > 0 else 0
                        with dcols[ki]:
                            st.markdown(
                                f'<div style="background:{colors[ki]}22;border:0.5px solid {colors[ki]};'
                                f'border-radius:6px;padding:8px;text-align:center;">'
                                f'<div style="font-size:16px;font-weight:600;color:{colors[ki]};">{pct:.1f}%</div>'
                                f'<div style="font-size:11px;color:#555;">Cluster {ki}</div>'
                                f'<div style="font-size:10px;color:#888;">{count:,} px</div></div>',
                                unsafe_allow_html=True
                            )

    # ── EXPORT ────────────────────────────────────────────────────────────────────
def array_to_geotiff_bytes(array, profile):
    profile = profile.copy()
    profile.update(dtype=rasterio.float32, count=1, nodata=-9999,
                   compress="lzw", driver="GTiff")
    
    # Add "interleave" and "photometric" to the keys being stripped
    for key in ["blockxsize", "blockysize", "tiled", "photometric", "interleave"]:
        profile.pop(key, None)
        
    buf = io.BytesIO()
    with rasterio.open(buf, "w", **profile) as dst:
        dst.write(np.where(np.isfinite(array), array, -9999).astype(np.float32), 1)
    buf.seek(0)
    return buf.read()


def cluster_to_geotiff_bytes(array, profile):
    profile = profile.copy()
    profile.update(dtype=rasterio.int32, count=1, nodata=-9999,
                   compress="lzw", driver="GTiff")
                   
    # Add "interleave" and "photometric" to the keys being stripped
    for key in ["blockxsize", "blockysize", "tiled", "photometric", "interleave"]:
        profile.pop(key, None)
        
    buf = io.BytesIO()
    with rasterio.open(buf, "w", **profile) as dst:
        dst.write(array.astype(np.int32), 1)
    buf.seek(0)
    return buf.read()

# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — FUSION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if not computed:
    st.stop()

st.markdown("---")
st.markdown("## Step 3 — Fusion analysis (optional)")
st.caption("Combines two indices to detect conditions single indices cannot reveal.")

fusion_status = {}
for fname, (i1, i2) in FUSION_REQUIRES.items():
    has_i1 = i1 in computed
    has_i2 = i2 in computed
    can_i1 = avail.get(i1, False)
    can_i2 = avail.get(i2, False)
    if has_i1 and has_i2:
        fusion_status[fname] = "ready"
    elif can_i1 and can_i2:
        fusion_status[fname] = "computable"
    else:
        missing_bands = []
        for idx in [i1, i2]:
            if not avail.get(idx, False):
                missing_bands += [b for b in INDEX_BANDS[idx] if b not in bands]
        fusion_status[fname] = ("missing_bands", list(set(missing_bands)))

for fname, status in fusion_status.items():
    i1, i2 = FUSION_REQUIRES[fname]
    if status == "ready":
        st.markdown(
            f'<span class="chip-available">✔ {fname}</span>',
            unsafe_allow_html=True
        )
    elif status == "computable":
        missing_computed = [x for x in [i1, i2] if x not in computed]
        st.markdown(
            f'<span class="chip-unavailable">⚠ {fname} — needs {", ".join(missing_computed)} to be computed</span>',
            unsafe_allow_html=True
        )
    else:
        _, mb = status
        st.markdown(
            f'<span class="chip-unavailable">✖ {fname} — missing bands: {", ".join(mb)}</span>',
            unsafe_allow_html=True
        )

st.markdown("")

for fname, status in fusion_status.items():
    if status == "computable":
        i1, i2 = FUSION_REQUIRES[fname]
        missing_computed = [x for x in [i1, i2] if x not in computed]
        for idx_needed in missing_computed:
            if st.button(f"➕ Compute {idx_needed} (required for {fname})",
                         key=f"ondemand_{fname}_{idx_needed}"):
                with st.spinner(f"Computing {idx_needed}..."):
                    bd  = st.session_state.bands
                    arr = _compute_index_cached(
                        idx_needed,
                        r=bd.get("red"),
                        nir=bd.get("nir"),
                        g=bd.get("green"),
                        b=bd.get("blue"),
                    )
                    st.session_state.computed_indices[idx_needed] = arr
                    st.session_state.cluster_results[idx_needed]  = _run_kmeans_cached(arr, k_val)
                st.success(f"{idx_needed} computed and stored.")
                st.rerun()

ready_fusions = [f for f, s in fusion_status.items() if s == "ready"]

if not ready_fusions:
    st.info(
        "No fusion modes are ready yet. Either compute the required indices above, "
        "or upload the missing bands."
    )
    st.stop()

fusion_mode = st.radio(
    "Select fusion mode",
    ready_fusions,
    label_visibility="collapsed",
)

st.markdown("")

i1, i2 = FUSION_REQUIRES[fusion_mode]
arr1    = st.session_state.computed_indices[i1]
arr2    = st.session_state.computed_indices[i2]

# ── (A) Water Stress ──────────────────────────────────────────────────────────
if fusion_mode == "NDVI + NDWI — Water stress":
    with st.expander("Adjust thresholds", expanded=False):
        fc1, fc2 = st.columns(2)
        with fc1:
            ndvi_t = st.slider("NDVI threshold", 0.0, 0.6, 0.3, 0.05,
                               help="Above = vegetation present")
        with fc2:
            ndwi_t = st.slider("NDWI threshold", -0.3, 0.3, 0.0, 0.05,
                               help="Above = sufficient water")

    zones = compute_water_stress_zones(arr1, arr2, ndvi_t, ndwi_t)
    # ✅ unique key
    st.plotly_chart(
        make_water_stress_fig(zones, arr1, arr2),
        width="stretch",
        key="fusion_water_stress",
    )
    st.markdown("##### Zone breakdown")
    st.markdown(zone_stats_html(zones, WATER_STRESS_ZONES), unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        "**How to read:** Green = healthy + well watered · Yellow = water-stressed (hidden) · "
        "Blue = waterlogged · Brown = bare/dry"
    )

# ── (B) Soil Interference ─────────────────────────────────────────────────────
elif fusion_mode == "NDVI + SAVI — Soil interference":
    with st.expander("Adjust threshold", expanded=False):
        pct_thresh = st.slider("Flag top N% as unreliable", 50, 95, 75, 5)

    diff_arr, unreliable, threshold = compute_soil_interference(arr1, arr2, pct_thresh)
    # ✅ unique key
    st.plotly_chart(
        make_soil_interference_fig(arr1, diff_arr, unreliable),
        width="stretch",
        key="fusion_soil_interference",
    )

    total_v = int(np.sum(np.isfinite(diff_arr)))
    u_ct    = int(np.sum(unreliable))
    r_ct    = total_v - u_ct
    pct_u   = u_ct / total_v * 100 if total_v > 0 else 0
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Reliable pixels",   f"{r_ct:,}")
    sc2.metric("Unreliable pixels", f"{u_ct:,}")
    sc3.metric("Unreliable %",       f"{pct_u:.1f}%")
    st.markdown("---")
    st.markdown(
        "**How to read:** Red overlay = pixels where `|NDVI − SAVI|` exceeds threshold. "
        "In those areas NDVI is inflated by soil — use SAVI values instead."
    )

# ── (C) Hidden Stress ─────────────────────────────────────────────────────────
elif fusion_mode == "NDVI + VARI — Hidden stress":
    with st.expander("Adjust thresholds", expanded=False):
        hc1, hc2 = st.columns(2)
        with hc1:
            vari_t  = st.slider("VARI threshold (visible greenness)", 0.0, 0.4, 0.1, 0.05)
        with hc2:
            ndvi_th = st.slider("NDVI threshold (NIR response)", 0.1, 0.6, 0.3, 0.05)

    hs_zones = compute_hidden_stress(arr1, arr2, vari_t, ndvi_th)
    # ✅ unique key
    st.plotly_chart(
        make_hidden_stress_fig(hs_zones, arr1, arr2),
        width="stretch",
        key="fusion_hidden_stress",
    )
    st.markdown("##### Zone breakdown")
    st.markdown(zone_stats_html(hs_zones, HIDDEN_STRESS_ZONES), unsafe_allow_html=True)

    hidden_ct  = int(np.sum(hs_zones == 1))
    total_veg  = int(np.sum(hs_zones >= 1))
    pct_hidden = hidden_ct / total_veg * 100 if total_veg > 0 else 0

    if pct_hidden > 15:
        st.error(
            f"**{pct_hidden:.1f}% of vegetated pixels show hidden stress.** "
            "Looks green visually but NIR response is poor — likely early physiological stress."
        )
    elif pct_hidden > 5:
        st.warning(f"**{pct_hidden:.1f}% hidden stress detected.** Monitor these zones closely.")
    else:
        st.success(f"Hidden stress is low ({pct_hidden:.1f}%). VARI and NDVI are in agreement.")

    st.markdown("---")
    st.markdown(
        "**How to read:** Red = hidden stress (high VARI, low NDVI) · "
        "Green = confirmed healthy · Grey = no anomaly / background"
    )