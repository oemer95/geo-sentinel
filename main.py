from sentinel_fetcher import get_time_series_data, load_config, setup_sh_config
from anomaly_detection import calculate_ndvi, detect_anomalies
from visualization import plot_ndvi_series
import numpy as np

def main():
    config_data = load_config()
    config = setup_sh_config(config_data["client_id"], config_data["client_secret"])

    print("Fetching Sentinel-2 time series...")
    raw_data = get_time_series_data(
        aoi=config_data["aoi"],
        time_range=tuple(config_data["time_range"]),
        bands=config_data["bands"],
        resolution=config_data["resolution"],
        config=config,
    )

    print("Processing NDVI...")
    ndvi_series = np.array([calculate_ndvi(scene) for scene in raw_data])

    print("Detecting anomalies...")
    anomalies = detect_anomalies(ndvi_series)

    print("Visualizing results...")
    plot_ndvi_series(ndvi_series, anomalies)

    print("Done")

if __name__ == "__main__":
    main()
