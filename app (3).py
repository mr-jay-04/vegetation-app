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

# ── CSS ──────────────────────────────────────────────────────────────────────
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
        font-size: 12px;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
INDEX_BANDS = {
    "NDVI": ["red", "nir"],
    "VARI": ["red", "green", "blue"],
    "NDWI": ["green", "nir"],
    "SAVI": ["red", "nir"],
}

INDEX_FORMULAS = {
    "NDVI": "( NIR − Red ) / ( NIR + Red )",
    "VARI": "( Green − Red ) / ( Green + Red − Blue )",
    "NDWI": "( Green − NIR ) / ( Green + NIR )",
    "SAVI": "( NIR − Red ) / ( NIR + Red + 0.5 )  ×  1.5",
}

HEALTH_THRESHOLDS = [
    (-1.0,  -0.1, "Water / Non-veg",  "#cfe2f3", "#1a5276", "Surface water or bare non-vegetated area."),
    (-0.1,   0.1, "Bare soil",        "#fdebd0", "#784212", "Exposed soil. Field may be fallow or pre-emergence."),
    ( 0.1,   0.2, "Very sparse",      "#fadbd8", "#922b21", "Very low density. Possible crop failure or early seedling stage."),
    ( 0.2,  0.35, "Sparse / Stressed","#fef9e7", "#7d6608", "Crop is stressed — possible water deficit or nutrient stress."),
    (0.35,   0.5, "Moderate",         "#eafaf1", "#1e8449", "Moderate vigour. Crop is growing but may need attention."),
    ( 0.5,  0.65, "Healthy",          "#d5f5e3", "#1a5e34", "Good canopy. Crop appears healthy with adequate resources."),
    ( 0.65,  1.0, "Very Healthy",     "#a9dfbf", "#0b3d25", "Dense vigorous canopy. Optimal conditions and high biomass."),
]

def get_health_label(val):
    for lo, hi, label, _, _, _ in HEALTH_THRESHOLDS:
        if lo <= val < hi:
            return label
    return "Very Healthy"

# ── HELPERS ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_band_bytes(file_bytes, band_num=1):
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            data = src.read(band_num).astype(np.float32)
            profile = src.profile.copy()
            meta = {
                "crs": str(src.crs),
                "transform": src.transform,
                "width": src.width,
                "height": src.height,
                "count": src.count,
            }
    return data, profile, meta

def read_band_from_file(uploaded_file, band_num=1):
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    return read_band_bytes(file_bytes, band_num)

def safe_divide(num, den, eps=1e-10):
    den_safe = np.where(np.abs(den) < eps, eps, den)
    result = num / den_safe
    result = np.where(np.abs(den) < eps, np.nan, result)
    return result

@st.cache_data(show_spinner=False)
def compute_index(index_name, r=None, nir=None, g=None, b=None):
    if index_name == "NDVI":
        return safe_divide(nir - r, nir + r)
    elif index_name == "VARI":
        return safe_divide(g - r, g + r - b)
    elif index_name == "NDWI":
        return safe_divide(g - nir, g + nir)
    elif index_name == "SAVI":
        L = 0.5
        return safe_divide(nir - r, nir + r + L) * (1 + L)

@st.cache_data(show_spinner=False)
def run_kmeans(index_array, k):
    flat = index_array.flatten()
    mask = np.isfinite(flat)
    valid = flat[mask].reshape(-1, 1)
    km = KMeans(n_clusters=k, random_state=42, n_init=3)
    labels = km.fit_predict(valid)
    result = np.full(flat.shape, -9999, dtype=np.int32)
    result[mask] = labels
    return result.reshape(index_array.shape)

# Keep display small — this is the key to preventing browser freeze
MAX_DISPLAY_PX = 500

def downsample(array, max_dim=MAX_DISPLAY_PX):
    h, w = array.shape
    if max(h, w) <= max_dim:
        return array
    scale = max_dim / max(h, w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    row_idx = np.linspace(0, h - 1, new_h, dtype=int)
    col_idx = np.linspace(0, w - 1, new_w, dtype=int)
    return array[np.ix_(row_idx, col_idx)]

def vectorized_health_labels(idx_ds):
    labels = np.full(idx_ds.shape, "Very Healthy", dtype=object)
    for lo, hi, label, _, _, _ in HEALTH_THRESHOLDS:
        mask = (idx_ds >= lo) & (idx_ds < hi)
        labels[mask] = label
    return labels

# ── PLOTLY MAPS (lightweight — only z + health label in hover) ────────────────
@st.cache_data(show_spinner=False)
def make_index_fig(index_array, index_name):
    ds = downsample(index_array)

    # Build health label array for hover
    health = vectorized_health_labels(ds)

    if index_name == "NDVI":
        colors = ["#1a6fa8", "#c8a45e", "#f5c518", "#8bc34a", "#4caf50", "#2e7d32", "#1b5e20"]
        colorscale = [[i / (len(colors) - 1), c] for i, c in enumerate(colors)]
    else:
        colorscale = "RdYlGn"

    zmin, zmax = (-1, 1) if index_name in ("NDVI", "NDWI") else (-0.5, 0.5)

    fig = go.Figure(go.Heatmap(
        z=ds,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        customdata=health,
        hovertemplate=(
            f"<b>{index_name}: %{{z:.4f}}</b><br>"
            "Health: %{customdata}<br>"
            "Pixel: (%{x}, %{y})<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text=index_name, side="right"),
            thickness=14,
            len=0.9,
        ),
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=480,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
        hoverlabel=dict(bgcolor="white", bordercolor="#ccc", font_size=13),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

@st.cache_data(show_spinner=False)
def make_cluster_fig(cluster_array, k):
    ds = downsample(cluster_array).astype(float)
    ds[ds == -9999] = np.nan

    cluster_colors = [
        "#1565c0", "#2e7d32", "#ef6c00",
        "#6a1b9a", "#c62828", "#00695c", "#f9a825", "#4e342e"
    ]
    colorscale = [
        [i / (k - 1) if k > 1 else 0, cluster_colors[i % len(cluster_colors)]]
        for i in range(k)
    ]

    fig = go.Figure(go.Heatmap(
        z=ds,
        colorscale=colorscale,
        zmin=0,
        zmax=k - 1,
        hovertemplate="<b>Cluster: %{z:.0f}</b><br>Pixel: (%{x}, %{y})<extra></extra>",
        colorbar=dict(
            title=dict(text="Cluster", side="right"),
            thickness=14,
            tickvals=list(range(k)),
            ticktext=[f"Cluster {i}" for i in range(k)],
            len=0.9,
        ),
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=480,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
        hoverlabel=dict(bgcolor="white", bordercolor="#ccc", font_size=13),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── FUSION ANALYSIS ───────────────────────────────────────────────────────────

# ── Shared layout helper ──────────────────────────────────────────────────────
def _base_layout(height=480):
    return dict(
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed", zeroline=False),
        hoverlabel=dict(bgcolor="white", bordercolor="#ccc", font_size=13),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

# ── (A) NDVI + NDWI — Water Stress Detection ─────────────────────────────────
# Classifies every pixel into 4 ecological zones based on two thresholds.
# NDVI threshold = 0.3  (below → low vegetation)
# NDWI threshold = 0.0  (below → low water content)

WATER_STRESS_ZONES = {
    0: ("Bare / Dry soil",          "#8B5E3C", "#fff"),   # low NDVI + low NDWI
    1: ("Waterlogged / Flooded",    "#2980B9", "#fff"),   # low NDVI + high NDWI
    2: ("Water-stressed veg",       "#F39C12", "#fff"),   # high NDVI + low NDWI  ← hidden stress
    3: ("Healthy vegetation",       "#27AE60", "#fff"),   # high NDVI + high NDWI
}

@st.cache_data(show_spinner=False)
def compute_water_stress_zones(ndvi_array, ndwi_array,
                                ndvi_thresh=0.3, ndwi_thresh=0.0):
    """
    Returns an int8 zone map:
      0 = Bare/Dry   1 = Waterlogged   2 = Water-stressed   3 = Healthy
    NaN where either input is NaN.
    """
    valid = np.isfinite(ndvi_array) & np.isfinite(ndwi_array)
    zones = np.full(ndvi_array.shape, -1, dtype=np.int8)

    hi_ndvi = ndvi_array >= ndvi_thresh
    hi_ndwi = ndwi_array >= ndwi_thresh

    zones[valid & ~hi_ndvi & ~hi_ndwi] = 0   # bare/dry
    zones[valid & ~hi_ndvi &  hi_ndwi] = 1   # waterlogged
    zones[valid &  hi_ndvi & ~hi_ndwi] = 2   # water-stressed
    zones[valid &  hi_ndvi &  hi_ndwi] = 3   # healthy
    return zones

@st.cache_data(show_spinner=False)
def make_water_stress_fig(zone_array, ndvi_ds, ndwi_ds):
    ds = downsample(zone_array.astype(float))
    ds_ndvi = downsample(ndvi_ds)
    ds_ndwi = downsample(ndwi_ds)
    ds[ds == -1] = np.nan

    # Build zone label array for hover
    zone_labels = np.full(ds.shape, "NoData", dtype=object)
    for zid, (label, _, _) in WATER_STRESS_ZONES.items():
        zone_labels[ds == zid] = label

    colorscale = [
        [0.00, WATER_STRESS_ZONES[0][1]],
        [0.33, WATER_STRESS_ZONES[0][1]],
        [0.33, WATER_STRESS_ZONES[1][1]],
        [0.66, WATER_STRESS_ZONES[1][1]],
        [0.66, WATER_STRESS_ZONES[2][1]],
        [0.99, WATER_STRESS_ZONES[2][1]],
        [0.99, WATER_STRESS_ZONES[3][1]],
        [1.00, WATER_STRESS_ZONES[3][1]],
    ]

    custom = np.stack([zone_labels, ds_ndvi, ds_ndwi], axis=-1)

    fig = go.Figure(go.Heatmap(
        z=ds,
        colorscale=colorscale,
        zmin=0, zmax=3,
        customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "NDVI: %{customdata[1]:.4f}<br>"
            "NDWI: %{customdata[2]:.4f}<br>"
            "Pixel: (%{x}, %{y})<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Zone", side="right"),
            thickness=14,
            tickvals=[0, 1, 2, 3],
            ticktext=[WATER_STRESS_ZONES[i][0] for i in range(4)],
            len=0.9,
        ),
        showscale=True,
    ))
    fig.update_layout(**_base_layout())
    return fig

# ── (B) NDVI + SAVI — Soil Interference Detection ────────────────────────────
# Where |NDVI - SAVI| is large, NDVI is being inflated/deflated by soil
# background — those pixels are unreliable for vegetation assessment.

@st.cache_data(show_spinner=False)
def compute_soil_interference(ndvi_array, savi_array):
    """
    Returns absolute difference array. High values = soil interference.
    Threshold at 75th percentile of valid differences to flag unreliable pixels.
    """
    diff = np.abs(ndvi_array - savi_array)
    valid_diff = diff[np.isfinite(diff)]
    threshold = np.nanpercentile(valid_diff, 75) if len(valid_diff) > 0 else 0.1
    unreliable_mask = (diff >= threshold) & np.isfinite(diff)
    return diff, unreliable_mask, float(threshold)

@st.cache_data(show_spinner=False)
def make_soil_interference_fig(ndvi_array, diff_array, unreliable_mask, threshold):
    ds_ndvi       = downsample(ndvi_array)
    ds_diff       = downsample(diff_array)
    ds_unreliable = downsample(unreliable_mask.astype(np.float32))

    # Base: NDVI map
    ndvi_colors = ["#1a6fa8","#c8a45e","#f5c518","#8bc34a","#4caf50","#2e7d32","#1b5e20"]
    ndvi_colorscale = [[i / (len(ndvi_colors) - 1), c] for i, c in enumerate(ndvi_colors)]

    # Reliability label per pixel
    reliability = np.where(ds_unreliable > 0.5, "Unreliable (soil interference)", "Reliable")
    custom = np.stack([reliability, ds_diff], axis=-1)

    fig = go.Figure()

    # Layer 1 — NDVI base heatmap
    fig.add_trace(go.Heatmap(
        z=ds_ndvi,
        colorscale=ndvi_colorscale,
        zmin=-1, zmax=1,
        customdata=custom,
        hovertemplate=(
            "<b>NDVI: %{z:.4f}</b><br>"
            "Reliability: %{customdata[0]}<br>"
            "|NDVI−SAVI|: %{customdata[1]:.4f}<br>"
            "Pixel: (%{x}, %{y})<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="NDVI", side="right"),
            thickness=14, len=0.9, x=1.02,
        ),
        showscale=True,
        name="NDVI",
    ))

    # Layer 2 — red semi-transparent overlay where unreliable
    # Build an RGBA overlay: red where unreliable, fully transparent elsewhere
    h, w = ds_unreliable.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    mask = ds_unreliable > 0.5
    overlay[mask]  = [220, 53, 69, 140]    # red with ~55% opacity
    overlay[~mask] = [0,   0,  0,   0]     # fully transparent

    import base64, PIL.Image
    pil_img = PIL.Image.fromarray(overlay, mode="RGBA")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    img_src = f"data:image/png;base64,{b64}"

    fig.add_layout_image(dict(
        source=img_src,
        xref="x", yref="y",
        x=0, y=0,
        sizex=w, sizey=h,
        sizing="stretch",
        opacity=1.0,
        layer="above",
    ))

    fig.update_layout(**_base_layout())
    fig.update_xaxes(range=[0, w])
    fig.update_yaxes(range=[h, 0])
    return fig

# ── (C) NDVI + VARI — Hidden Stress Detection ────────────────────────────────
# Anomaly: high VARI (looks green visually) but low NDVI (poor NIR response)
# This reveals vegetation that appears healthy to the eye but is physiologically stressed.

@st.cache_data(show_spinner=False)
def compute_hidden_stress(ndvi_array, vari_array,
                           vari_thresh=0.1, ndvi_thresh=0.3):
    """
    Returns a zone map:
      0 = Normal / no anomaly
      1 = Hidden stress  (high VARI + low NDVI)
      2 = Confirmed healthy (high VARI + high NDVI)
    NaN where either input is NaN.
    """
    valid = np.isfinite(ndvi_array) & np.isfinite(vari_array)
    zones = np.full(ndvi_array.shape, -1, dtype=np.int8)

    hi_vari = vari_array >= vari_thresh
    hi_ndvi = ndvi_array >= ndvi_thresh

    zones[valid & ~hi_vari & ~hi_ndvi] = 0   # normal background
    zones[valid &  hi_vari & ~hi_ndvi] = 1   # hidden stress ← anomaly
    zones[valid &  hi_vari &  hi_ndvi] = 2   # confirmed healthy
    zones[valid & ~hi_vari &  hi_ndvi] = 0   # strong NIR, low visible green — normal
    return zones

HIDDEN_STRESS_ZONES = {
    0: ("No anomaly",         "#B2BABB"),
    1: ("Hidden stress",      "#E74C3C"),   # the key anomaly zone — red
    2: ("Confirmed healthy",  "#27AE60"),
}

@st.cache_data(show_spinner=False)
def make_hidden_stress_fig(zone_array, ndvi_ds, vari_ds):
    ds      = downsample(zone_array.astype(float))
    ds_ndvi = downsample(ndvi_ds)
    ds_vari = downsample(vari_ds)
    ds[ds == -1] = np.nan

    zone_labels = np.full(ds.shape, "NoData", dtype=object)
    for zid, (label, _) in HIDDEN_STRESS_ZONES.items():
        zone_labels[ds == zid] = label

    colorscale = [
        [0.00, HIDDEN_STRESS_ZONES[0][1]],
        [0.33, HIDDEN_STRESS_ZONES[0][1]],
        [0.33, HIDDEN_STRESS_ZONES[1][1]],
        [0.66, HIDDEN_STRESS_ZONES[1][1]],
        [0.66, HIDDEN_STRESS_ZONES[2][1]],
        [1.00, HIDDEN_STRESS_ZONES[2][1]],
    ]

    custom = np.stack([zone_labels, ds_ndvi, ds_vari], axis=-1)

    fig = go.Figure(go.Heatmap(
        z=ds,
        colorscale=colorscale,
        zmin=0, zmax=2,
        customdata=custom,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "NDVI: %{customdata[1]:.4f}<br>"
            "VARI: %{customdata[2]:.4f}<br>"
            "Pixel: (%{x}, %{y})<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Zone", side="right"),
            thickness=14,
            tickvals=[0, 1, 2],
            ticktext=[HIDDEN_STRESS_ZONES[i][0] for i in range(3)],
            len=0.9,
        ),
        showscale=True,
    ))
    fig.update_layout(**_base_layout())
    return fig

# ── Fusion zone stats helper ──────────────────────────────────────────────────
def zone_stats_html(zone_array, zone_dict, nodata_val=-1):
    """Returns HTML summary cards for each zone — pixel count + percentage."""
    total = np.sum(zone_array != nodata_val)
    cards = ""
    for zid, info in zone_dict.items():
        color = info[1]
        label = info[0]
        count = int(np.sum(zone_array == zid))
        pct   = count / total * 100 if total > 0 else 0
        cards += (
            f'<div style="flex:1;min-width:130px;background:{color}22;border:0.5px solid {color};'
            f'border-radius:8px;padding:10px;text-align:center;">'
            f'<div style="font-size:17px;font-weight:600;color:{color};">{pct:.1f}%</div>'
            f'<div style="font-size:12px;color:#444;margin-top:2px;">{label}</div>'
            f'<div style="font-size:10px;color:#888;">{count:,} px</div>'
            f'</div>'
        )
    return f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:8px;">{cards}</div>'

# ── EXPORT ────────────────────────────────────────────────────────────────────
def array_to_geotiff_bytes(array, profile):
    profile = profile.copy()
    profile.update(dtype=rasterio.float32, count=1, nodata=-9999, compress="lzw", driver="GTiff")
    for key in ["blockxsize", "blockysize", "tiled"]:
        profile.pop(key, None)
    buf = io.BytesIO()
    with rasterio.open(buf, "w", **profile) as dst:
        out = np.where(np.isfinite(array), array, -9999).astype(np.float32)
        dst.write(out, 1)
    buf.seek(0)
    return buf.read()

def cluster_to_geotiff_bytes(array, profile):
    profile = profile.copy()
    profile.update(dtype=rasterio.int32, count=1, nodata=-9999, compress="lzw", driver="GTiff")
    for key in ["blockxsize", "blockysize", "tiled"]:
        profile.pop(key, None)
    buf = io.BytesIO()
    with rasterio.open(buf, "w", **profile) as dst:
        dst.write(array.astype(np.int32), 1)
    buf.seek(0)
    return buf.read()

# ── SESSION STATE ─────────────────────────────────────────────────────────────
for key in ["index_array", "cluster_array", "profile", "bands_used", "index_name",
            "all_bands"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Vegetation Analyser")
    st.markdown("---")
    st.markdown('<div class="section-title">Upload mode</div>', unsafe_allow_html=True)
    upload_mode = st.radio(
        "Upload mode",
        ["Multispectral (single file)", "Single bands (individual files)", "RGB combined (VARI only)"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown('<div class="section-title">Vegetation index</div>', unsafe_allow_html=True)
    selected_index = st.radio("Index", ["NDVI", "VARI", "NDWI", "SAVI"], label_visibility="collapsed")
    st.caption(f"Formula: `{INDEX_FORMULAS[selected_index]}`")
    st.markdown("---")
    st.markdown('<div class="section-title">K-Means clusters</div>', unsafe_allow_html=True)
    k_val = st.slider("Number of clusters (K)", min_value=2, max_value=8, value=3, step=1)
    st.markdown("---")
    run_btn = st.button("Run analysis", type="primary", use_container_width=True)

# ── MAIN CONTENT ──────────────────────────────────────────────────────────────
st.markdown("### Upload your imagery")

bands = {}
profile_ref = None
error_msg = None

if upload_mode == "RGB combined (VARI only)" and selected_index != "VARI":
    st.error(
        f"**RGB images cannot calculate {selected_index}.** "
        f"Switch the index to **VARI**, or upload a multispectral file for {selected_index}."
    )

elif upload_mode == "Multispectral (single file)":
    ms_file = st.file_uploader("Upload multispectral GeoTIFF", type=["tif", "tiff"], key="ms_file")
    if ms_file:
        _, ref_profile, meta = read_band_from_file(ms_file, 1)
        profile_ref = ref_profile
        st.info(f"File loaded — {meta['count']} band(s) | {meta['width']}×{meta['height']} px | CRS: {meta['crs']}")
        needed_bands = INDEX_BANDS[selected_index]
        col_list = st.columns(len(needed_bands))
        band_options = [f"Band {i+1}" for i in range(meta["count"])]
        for i, bname in enumerate(needed_bands):
            with col_list[i]:
                choice = st.selectbox(
                    f"Which band is **{bname.upper()}**?",
                    options=band_options,
                    index=min(i, meta["count"] - 1),
                    key=f"band_sel_{bname}"
                )
                band_num = int(choice.split(" ")[1])
                arr, _, _ = read_band_from_file(ms_file, band_num)
                bands[bname] = arr

elif upload_mode == "Single bands (individual files)":
    needed_bands = INDEX_BANDS[selected_index]
    cols = st.columns(len(needed_bands))
    for i, bname in enumerate(needed_bands):
        with cols[i]:
            f = st.file_uploader(f"{bname.upper()} band (.tif)", type=["tif", "tiff"], key=f"single_{bname}")
            if f:
                arr, prof, meta = read_band_from_file(f, 1)
                bands[bname] = arr
                if profile_ref is None:
                    profile_ref = prof
                st.caption(f"{meta['width']}×{meta['height']} px")
    if len(bands) == len(needed_bands):
        shapes = [v.shape for v in bands.values()]
        if len(set(shapes)) > 1:
            error_msg = "All band files must have the same dimensions. Shapes: " + str(shapes)

elif upload_mode == "RGB combined (VARI only)":
    rgb_file = st.file_uploader("Upload RGB image", type=["tif", "tiff", "jpg", "jpeg", "png"], key="rgb_file")
    if rgb_file:
        r_arr, prof, meta = read_band_from_file(rgb_file, 1)
        g_arr, _, _       = read_band_from_file(rgb_file, 2)
        b_arr, _, _       = read_band_from_file(rgb_file, 3)
        bands = {"red": r_arr, "green": g_arr, "blue": b_arr}
        profile_ref = prof
        st.info(f"RGB loaded — {meta['width']}×{meta['height']} px | R=1, G=2, B=3")

if error_msg:
    st.error(error_msg)

needed = INDEX_BANDS[selected_index]
missing = [b for b in needed if b not in bands]

if bands and missing:
    st.error(
        f"**{selected_index} needs: {', '.join(b.upper() for b in needed)}** — "
        f"Missing: {', '.join(b.upper() for b in missing)}."
    )

# ── RUN ANALYSIS ──────────────────────────────────────────────────────────────
if run_btn:
    if not bands or missing:
        st.error("Please upload all required bands before running.")
    elif not error_msg:
        progress_bar = st.progress(0, text="Starting analysis...")
        status = st.empty()

        progress_bar.progress(10, text="Step 1 / 4 — Checking band values...")
        status.caption("Step 1 / 4 — Checking band values...")
        if "red" in bands and "nir" in bands:
            if np.nanmean(bands["red"]) > np.nanmean(bands["nir"]):
                st.warning("Red band mean > NIR mean — bands may be swapped. Check assignments.")

        progress_bar.progress(25, text=f"Step 2 / 4 — Computing {selected_index}...")
        status.caption(f"Step 2 / 4 — Computing {selected_index}...")
        idx_arr = compute_index(
            selected_index,
            r=bands.get("red"),
            nir=bands.get("nir"),
            g=bands.get("green"),
            b=bands.get("blue"),
        )

        progress_bar.progress(55, text=f"Step 3 / 4 — Running K-Means (K={k_val})...")
        status.caption(f"Step 3 / 4 — Running K-Means (K={k_val})...")
        cl_arr = run_kmeans(idx_arr, k_val)

        progress_bar.progress(80, text="Step 4 / 4 — Preparing maps...")
        status.caption("Step 4 / 4 — Preparing maps...")
        st.session_state.index_array   = idx_arr
        st.session_state.cluster_array = cl_arr
        st.session_state.profile       = profile_ref
        st.session_state.bands_used    = bands
        st.session_state.index_name    = selected_index
        st.session_state.all_bands     = bands   # kept for fusion analysis

        progress_bar.progress(100, text="Done!")
        status.empty()
        progress_bar.empty()

# ── OUTPUT ────────────────────────────────────────────────────────────────────
if st.session_state.index_array is not None:
    idx_arr    = st.session_state.index_array
    cl_arr     = st.session_state.cluster_array
    idx_name   = st.session_state.index_name
    bands_used = st.session_state.bands_used

    st.markdown("---")
    st.markdown("### Results")

    valid_vals = idx_arr[np.isfinite(idx_arr)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Max {idx_name}",  f"{np.nanmax(valid_vals):.4f}")
    c2.metric(f"Mean {idx_name}", f"{np.nanmean(valid_vals):.4f}")
    c3.metric(f"Min {idx_name}",  f"{np.nanmin(valid_vals):.4f}")
    c4.metric("Valid pixels",     f"{len(valid_vals):,}")

    st.markdown("---")
    tab1, tab2 = st.tabs([f"{idx_name} Map", "Cluster Map"])

    with tab1:
        st.markdown(f"**Hover over the map** to see {idx_name} value and crop health at each pixel.")
        st.plotly_chart(make_index_fig(idx_arr, idx_name), use_container_width=True)

        st.markdown("##### Crop health scale")
        cols = st.columns(len(HEALTH_THRESHOLDS))
        for i, (lo, hi, label, bg, tc, desc) in enumerate(HEALTH_THRESHOLDS):
            with cols[i]:
                st.markdown(
                    f'<div style="background:{bg};color:{tc};padding:6px 8px;border-radius:6px;'
                    f'font-size:11px;text-align:center;font-weight:600;">'
                    f'{label}<br><span style="font-weight:400;">{lo} to {hi}</span></div>',
                    unsafe_allow_html=True
                )

    with tab2:
        st.markdown(f"**Hover over the map** to see cluster number at each pixel.")
        st.plotly_chart(make_cluster_fig(cl_arr, k_val), use_container_width=True)

        st.markdown("##### Cluster distribution")
        dist_cols = st.columns(k_val)
        cluster_colors = ["#1565c0","#2e7d32","#ef6c00","#6a1b9a","#c62828","#00695c","#f9a825","#4e342e"]
        total_valid = np.sum(cl_arr != -9999)
        for ki in range(k_val):
            count = np.sum(cl_arr == ki)
            pct = count / total_valid * 100 if total_valid > 0 else 0
            with dist_cols[ki]:
                st.markdown(
                    f'<div style="background:{cluster_colors[ki]}22;border:0.5px solid {cluster_colors[ki]};'
                    f'border-radius:6px;padding:8px;text-align:center;">'
                    f'<div style="font-size:18px;font-weight:600;color:{cluster_colors[ki]};">{pct:.1f}%</div>'
                    f'<div style="font-size:11px;color:#555;">Cluster {ki}</div>'
                    f'<div style="font-size:10px;color:#888;">{count:,} px</div></div>',
                    unsafe_allow_html=True
                )

    st.markdown("---")
    st.markdown("### Export")
    dl1, dl2 = st.columns(2)
    if st.session_state.profile:
        idx_bytes = array_to_geotiff_bytes(idx_arr, st.session_state.profile)
        cl_bytes  = cluster_to_geotiff_bytes(cl_arr, st.session_state.profile)
        with dl1:
            st.download_button(
                label=f"Download {idx_name} GeoTIFF",
                data=idx_bytes,
                file_name=f"{idx_name.lower()}_output.tif",
                mime="image/tiff",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                label="Download Cluster GeoTIFF",
                data=cl_bytes,
                file_name="cluster_output.tif",
                mime="image/tiff",
                use_container_width=True,
            )
    else:
        st.info("GeoTIFF export requires geospatial metadata (CRS + transform) in the input file.")

# ── FUSION ANALYSIS SECTION ───────────────────────────────────────────────────
if st.session_state.index_array is not None:
    stored_bands = st.session_state.all_bands or {}

    # Determine which fusion modes are possible given available bands
    has_nir   = "nir"   in stored_bands
    has_green = "green" in stored_bands
    has_red   = "red"   in stored_bands
    has_blue  = "blue"  in stored_bands

    can_water_stress = has_nir and has_green and has_red   # needs NDVI + NDWI
    can_soil_interf  = has_nir and has_red                  # needs NDVI + SAVI
    can_hidden_stress= has_nir and has_red and has_green and has_blue  # needs NDVI + VARI

    if not any([can_water_stress, can_soil_interf, can_hidden_stress]):
        st.info(
            "Fusion analysis requires additional bands. "
            "Upload NIR + Red + Green + Blue for all three fusion modes."
        )
    else:
        st.markdown("---")
        st.markdown("### Fusion analysis")
        st.caption(
            "Combines two indices to detect crop conditions that single indices cannot reveal. "
            "Select a mode below."
        )

        # Build list of available fusion modes dynamically
        available_modes = []
        if can_water_stress:
            available_modes.append("NDVI + NDWI — Water stress detection")
        if can_soil_interf:
            available_modes.append("NDVI + SAVI — Soil interference detection")
        if can_hidden_stress:
            available_modes.append("NDVI + VARI — Hidden stress detection")

        fusion_mode = st.radio(
            "Fusion mode",
            available_modes,
            horizontal=True,
            label_visibility="collapsed",
        )

        # ── Shared threshold sliders ──────────────────────────────────────────
        r   = stored_bands.get("red")
        nir = stored_bands.get("nir")
        g   = stored_bands.get("green")
        b   = stored_bands.get("blue")

        # ── (A) Water Stress ─────────────────────────────────────────────────
        if fusion_mode == "NDVI + NDWI — Water stress detection":
            with st.expander("Adjust thresholds", expanded=False):
                tc1, tc2 = st.columns(2)
                with tc1:
                    ndvi_thresh = st.slider(
                        "NDVI threshold (vegetation / no vegetation)",
                        min_value=0.0, max_value=0.6, value=0.3, step=0.05,
                        help="Pixels above this = vegetation present"
                    )
                with tc2:
                    ndwi_thresh = st.slider(
                        "NDWI threshold (water content)",
                        min_value=-0.3, max_value=0.3, value=0.0, step=0.05,
                        help="Pixels above this = sufficient water content"
                    )

            with st.spinner("Computing NDVI + NDWI fusion..."):
                ndvi_arr = compute_index("NDVI", r=r, nir=nir)
                ndwi_arr = compute_index("NDWI", r=r, nir=nir, g=g)
                zones    = compute_water_stress_zones(ndvi_arr, ndwi_arr,
                                                      ndvi_thresh, ndwi_thresh)

            st.plotly_chart(
                make_water_stress_fig(zones, ndvi_arr, ndwi_arr),
                use_container_width=True
            )

            # Zone legend + stats
            st.markdown("##### Zone breakdown")
            st.markdown(
                zone_stats_html(zones, WATER_STRESS_ZONES),
                unsafe_allow_html=True
            )

            st.markdown("---")
            st.markdown(
                "**How to read this map:**  \n"
                "- 🟢 **Healthy** — good vegetation density and water content  \n"
                "- 🟡 **Water-stressed** — crop looks dense but is moisture-deficient (hidden stress)  \n"
                "- 🔵 **Waterlogged** — excess water, possible flooding or poor drainage  \n"
                "- 🟫 **Bare/Dry** — sparse or no vegetation and dry conditions"
            )

        # ── (B) Soil Interference ────────────────────────────────────────────
        elif fusion_mode == "NDVI + SAVI — Soil interference detection":
            with st.expander("Adjust threshold", expanded=False):
                percentile_thresh = st.slider(
                    "Flag top N% of pixels as unreliable",
                    min_value=50, max_value=95, value=75, step=5,
                    help="Higher = stricter, fewer pixels flagged"
                )

            with st.spinner("Computing NDVI + SAVI fusion..."):
                ndvi_arr  = compute_index("NDVI", r=r, nir=nir)
                savi_arr  = compute_index("SAVI", r=r, nir=nir)
                diff_arr, unreliable_mask, threshold = compute_soil_interference(
                    ndvi_arr, savi_arr
                )
                # recompute with user percentile
                valid_diff = diff_arr[np.isfinite(diff_arr)]
                threshold  = float(np.nanpercentile(valid_diff, percentile_thresh))
                unreliable_mask = (diff_arr >= threshold) & np.isfinite(diff_arr)

            try:
                fig_soil = make_soil_interference_fig(
                    ndvi_arr, diff_arr, unreliable_mask, threshold
                )
                st.plotly_chart(fig_soil, use_container_width=True)
            except ImportError:
                st.warning(
                    "PIL (Pillow) is required for the red overlay. "
                    "Run `pip install Pillow` and restart the app. "
                    "Showing plain NDVI map instead."
                )
                st.plotly_chart(make_index_fig(ndvi_arr, "NDVI"), use_container_width=True)

            # Stats
            total_valid   = int(np.sum(np.isfinite(diff_arr)))
            unreliable_ct = int(np.sum(unreliable_mask))
            reliable_ct   = total_valid - unreliable_ct
            pct_unrel     = unreliable_ct / total_valid * 100 if total_valid > 0 else 0

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Reliable pixels",    f"{reliable_ct:,}")
            sc2.metric("Unreliable pixels",  f"{unreliable_ct:,}")
            sc3.metric("Unreliable %",        f"{pct_unrel:.1f}%")

            st.markdown("---")
            st.markdown(
                "**How to read this map:**  \n"
                "Red overlay marks pixels where `|NDVI − SAVI|` exceeds the threshold — "
                "these are areas where partial soil exposure is inflating or deflating the NDVI reading. "
                "Use SAVI values instead of NDVI for those regions."
            )

        # ── (C) Hidden Stress ────────────────────────────────────────────────
        elif fusion_mode == "NDVI + VARI — Hidden stress detection":
            with st.expander("Adjust thresholds", expanded=False):
                hc1, hc2 = st.columns(2)
                with hc1:
                    vari_thresh = st.slider(
                        "VARI threshold (visible greenness)",
                        min_value=0.0, max_value=0.4, value=0.1, step=0.05,
                        help="Pixels above this = visually green"
                    )
                with hc2:
                    ndvi_thresh_h = st.slider(
                        "NDVI threshold (NIR response)",
                        min_value=0.1, max_value=0.6, value=0.3, step=0.05,
                        help="Pixels below this = low NIR activity"
                    )

            with st.spinner("Computing NDVI + VARI fusion..."):
                ndvi_arr  = compute_index("NDVI", r=r, nir=nir)
                vari_arr  = compute_index("VARI", r=r, g=g, b=b)
                hs_zones  = compute_hidden_stress(
                    ndvi_arr, vari_arr, vari_thresh, ndvi_thresh_h
                )

            st.plotly_chart(
                make_hidden_stress_fig(hs_zones, ndvi_arr, vari_arr),
                use_container_width=True
            )

            # Zone stats
            st.markdown("##### Zone breakdown")
            st.markdown(
                zone_stats_html(hs_zones, HIDDEN_STRESS_ZONES),
                unsafe_allow_html=True
            )

            # Highlight hidden stress specifically
            hidden_ct  = int(np.sum(hs_zones == 1))
            total_veg  = int(np.sum(hs_zones >= 1))
            pct_hidden = hidden_ct / total_veg * 100 if total_veg > 0 else 0

            if pct_hidden > 15:
                st.error(
                    f"**{pct_hidden:.1f}% of vegetated pixels show hidden stress.** "
                    "These areas look green to the eye but have poor NIR response — "
                    "likely early-stage physiological stress not yet visible."
                )
            elif pct_hidden > 5:
                st.warning(
                    f"**{pct_hidden:.1f}% of vegetated pixels show hidden stress.** "
                    "Monitor these zones closely."
                )
            else:
                st.success(
                    f"Hidden stress is low ({pct_hidden:.1f}% of vegetated pixels). "
                    "VARI and NDVI are largely in agreement."
                )

            st.markdown("---")
            st.markdown(
                "**How to read this map:**  \n"
                "- 🔴 **Hidden stress** — vegetation looks green (high VARI) "
                "but NIR response is poor (low NDVI). Physiological stress not yet visible to the eye.  \n"
                "- 🟢 **Confirmed healthy** — both visible greenness and NIR response are strong  \n"
                "- ⬜ **No anomaly** — background, soil, or water areas"
            )