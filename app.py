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

# ── HELPERS ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_band_bytes(file_bytes, band_num=1):
    """Cache by raw bytes — works correctly with Streamlit's cache."""
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
    """Read file bytes once, then use cached function."""
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
    """Each band passed as separate array — avoids dict caching issue."""
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

MAX_DISPLAY_PX = 800

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
    labels = np.full(idx_ds.shape, "NoData", dtype=object)
    for lo, hi, label, _, _, _ in HEALTH_THRESHOLDS:
        mask = (idx_ds >= lo) & (idx_ds < hi)
        labels[mask] = label
    return labels

def build_hover_arrays(index_array, bands, cluster_array):
    idx_ds = downsample(index_array)
    cl_ds  = downsample(cluster_array)
    health_labels = vectorized_health_labels(idx_ds)
    custom = [health_labels, cl_ds.astype(str)]
    band_names = []
    for bn, arr in bands.items():
        if arr is not None:
            custom.append(downsample(arr))
            band_names.append(bn.upper())
    return idx_ds, np.stack(custom, axis=-1), band_names

def ndvi_health_colorscale():
    return [
        [0.0,  "#1a6fa8"],
        [0.27, "#c8a45e"],
        [0.37, "#f5c518"],
        [0.50, "#8bc34a"],
        [0.65, "#4caf50"],
        [0.82, "#2e7d32"],
        [1.0,  "#1b5e20"],
    ]

def make_plotly_map(index_array, cluster_array, bands, index_name):
    idx_ds, custom, band_names = build_hover_arrays(index_array, bands, cluster_array)
    extra_tmpl = "".join(
        f"{bn}: %{{customdata[{i+2}]:.3f}}<br>"
        for i, bn in enumerate(band_names)
    )
    hover_tmpl = (
        f"<b>{index_name}: %{{z:.4f}}</b><br>"
        "Health: %{customdata[0]}<br>"
        "Cluster: %{customdata[1]}<br>"
        + extra_tmpl +
        "Pixel: (%{x}, %{y})<extra></extra>"
    )
    colorscale = ndvi_health_colorscale() if index_name == "NDVI" else "RdYlGn"
    zmin, zmax = (-1, 1) if index_name in ("NDVI", "NDWI") else (-0.5, 0.5)
    fig = go.Figure(go.Heatmap(
        z=idx_ds,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        customdata=custom,
        hovertemplate=hover_tmpl,
        colorbar=dict(title=dict(text=index_name, side="right"), thickness=14, len=0.9),
        showscale=True,
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=480,
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
        hoverlabel=dict(bgcolor="white", bordercolor="#cccccc", font_size=13),
        hovermode="closest",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

def make_cluster_fig(cluster_array, k):
    display = downsample(cluster_array).astype(float)
    display[display == -9999] = np.nan
    cluster_colors = [
        "#1565c0", "#2e7d32", "#ef6c00",
        "#6a1b9a", "#c62828", "#00695c", "#f9a825", "#4e342e"
    ]
    if k == 1:
        colorscale = [[0.0, cluster_colors[0]], [1.0, cluster_colors[0]]]
    else:
        colorscale = [[i / (k - 1), cluster_colors[i % len(cluster_colors)]] for i in range(k)]
    fig = go.Figure(go.Contour(
        z=display,
        colorscale=colorscale,
        zmin=0, zmax=k - 1,
        ncontours=k * 6,
        contours=dict(coloring="fill", showlines=False),
        hovertemplate="Cluster: %{z:.2f}<br>Pixel: (%{x}, %{y})<extra></extra>",
        colorbar=dict(
            title=dict(text="Cluster", side="right"),
            thickness=14,
            tickvals=list(range(k)),
            ticktext=[f"Cluster {i}" for i in range(k)],
            len=0.9,
        ),
        line_smoothing=1.3,
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=480,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed", zeroline=False),
        hoverlabel=dict(bgcolor="white", bordercolor="#ccc", font_size=13),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

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
for key in ["index_array", "cluster_array", "profile", "bands_used", "index_name"]:
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

        # Step 1 — band swap check
        progress_bar.progress(10, text="Step 1 / 4 — Checking band values...")
        status.caption("Step 1 / 4 — Checking band values...")
        if "red" in bands and "nir" in bands:
            if np.nanmean(bands["red"]) > np.nanmean(bands["nir"]):
                st.warning("Red band mean > NIR mean — bands may be swapped. Check assignments.")

        # Step 2 — compute index (pass arrays individually for proper caching)
        progress_bar.progress(25, text=f"Step 2 / 4 — Computing {selected_index}...")
        status.caption(f"Step 2 / 4 — Computing {selected_index}...")
        idx_arr = compute_index(
            selected_index,
            r=bands.get("red"),
            nir=bands.get("nir"),
            g=bands.get("green"),
            b=bands.get("blue"),
        )

        # Step 3 — clustering
        progress_bar.progress(55, text=f"Step 3 / 4 — Running K-Means (K={k_val})...")
        status.caption(f"Step 3 / 4 — Running K-Means (K={k_val})...")
        cl_arr = run_kmeans(idx_arr, k_val)

        # Step 4 — store results
        progress_bar.progress(80, text="Step 4 / 4 — Preparing display...")
        status.caption("Step 4 / 4 — Preparing display...")
        st.session_state.index_array   = idx_arr
        st.session_state.cluster_array = cl_arr
        st.session_state.profile       = profile_ref
        st.session_state.bands_used    = bands
        st.session_state.index_name    = selected_index

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
        st.markdown(f"**Hover over the map** to see {idx_name} value, health, band values and cluster.")
        fig_idx = make_plotly_map(idx_arr, cl_arr, bands_used, idx_name)
        st.plotly_chart(fig_idx, use_container_width=True)

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
        st.markdown(f"**K = {k_val} clusters** — hover to see cluster at each pixel.")
        fig_cl = make_cluster_fig(cl_arr, k_val)
        st.plotly_chart(fig_cl, use_container_width=True)

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