# GeoSentinel: Anomaly Detection in Sentinel-2 Time Series

GeoSentinel is a Python-based project designed to detect and visualize anomalies in remote sensing data, specifically from the Sentinel-2 satellite. This tool is valuable for environmental monitoring, change detection, disaster response, and land use tracking.

---

## Features
- Download Sentinel-2 time series using the SentinelHub API
- Analyze multi-band data over time for a region of interest (ROI)
- Apply statistical and machine learning methods (e.g., Isolation Forest) for anomaly detection
- Visualize time series and anomaly maps

---
## Project Structure
```
GeoSentinel/
├── sentinel_fetcher.py       # Downloads Sentinel-2 data from SentinelHub
├── anomaly_detection.py      # Detects anomalies in the NDVI or reflectance series
├── visualization.py          # Visualization of time series and anomaly maps
├── config.yaml               # Config file with ROI, time range, bands, credentials
├── main.py                   # Entry point to run the full pipeline
└── README.md                 # Documentation
```

---

## Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```
Dependencies include:
- numpy, pandas, matplotlib, seaborn
- rasterio, geopandas
- scikit-learn
- sentinelhub

---

## Usage
1. Create an account at [Sentinel Hub](https://www.sentinel-hub.com/) and obtain your credentials.
2. Fill out `config.yaml` with your AOI, bands, time range, and credentials.
3. Run the pipeline:
```bash
python main.py
```
This will:
- Download and preprocess the time series
- Run anomaly detection
- Output maps and time series plots in the `output/` folder

---

## Example Output
- NDVI time series with detected outliers
- RGB image overlays with anomaly locations

---

## Potential Extensions
- Integrate with deep learning (e.g. LSTM) for sequence-based detection
- Add land cover classification support
- Streamline via Web UI or dashboard

---

## License
MIT License
For research or environmental monitoring applications. Not affiliated with ESA or Sentinel Hub.
