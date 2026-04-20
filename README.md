# Vegetation Analyser

A Streamlit web app for vegetation and crop health monitoring using NDVI, VARI, NDWI, and SAVI indices with K-Means clustering.

## Features
- Three upload modes: Multispectral, Single bands, RGB combined
- Indices: NDVI, VARI, NDWI, SAVI
- K-Means clustering (K = 2–8)
- Interactive Plotly map — hover to see index value, crop health, band values, cluster
- Crop health diagnosis based on NDVI thresholds
- Export NDVI and cluster maps as GeoTIFF

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Click "New app"
4. Select your repo, branch, and set main file path to `app.py`
5. Click Deploy

## Folder structure

```
vegetation_app/
├── app.py
├── requirements.txt
└── .streamlit/
    └── config.toml
```

## Index band requirements

| Index | Bands needed |
|-------|-------------|
| NDVI  | Red, NIR |
| VARI  | Red, Green, Blue |
| NDWI  | Green, NIR |
| SAVI  | Red, NIR |

## Crop health scale (NDVI-based)

| NDVI range | Label |
|------------|-------|
| < −0.1 | Water / Non-veg |
| −0.1 to 0.1 | Bare soil |
| 0.1 to 0.2 | Very sparse |
| 0.2 to 0.35 | Sparse / Stressed |
| 0.35 to 0.5 | Moderate |
| 0.5 to 0.65 | Healthy |
| > 0.65 | Very healthy |
