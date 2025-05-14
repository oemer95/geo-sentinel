# anomaly_detection.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import seaborn as sns
from datetime import datetime
import os

def calculate_ndvi(data):
    """Calculate Normalized Difference Vegetation Index from multispectral imagery."""
    nir = data[:, :, 7]
    red = data[:, :, 3]
    ndvi = (nir - red) / (nir + red + 1e-6)
    return ndvi

def calculate_evi(data):
    """Calculate Enhanced Vegetation Index from multispectral imagery."""
    nir = data[:, :, 7]
    red = data[:, :, 3]
    blue = data[:, :, 1]
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))
    return evi

def calculate_savi(data, L=0.5):
    """Calculate Soil-Adjusted Vegetation Index from multispectral imagery."""
    nir = data[:, :, 7]
    red = data[:, :, 3]
    savi = ((nir - red) / (nir + red + L)) * (1 + L)
    return savi

def detect_anomalies(ndvi_series, method='iforest', contamination=0.05):
    """
    Detect anomalies in vegetation indices time series using various methods.
    
    Parameters:
    -----------
    ndvi_series : numpy.ndarray
        Time series of vegetation indices (e.g., NDVI, EVI)
    method : str
        Method to use for anomaly detection ('iforest', 'dbscan', or 'ensemble')
    contamination : float
        Expected proportion of outliers in the dataset (for Isolation Forest)
        
    Returns:
    --------
    anomalies : numpy.ndarray
        Array of labels where -1 indicates anomalies
    scores : numpy.ndarray
        Anomaly scores (only for Isolation Forest)
    """
    # Reshape the data for processing
    flattened = ndvi_series.reshape(ndvi_series.shape[0], -1)
    df = pd.DataFrame(flattened)
    
    # Standardize the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    if method == 'iforest':
        # Isolation Forest method
        clf = IsolationForest(
            contamination=contamination, 
            random_state=42, 
            n_estimators=100,
            max_samples='auto'
        )
        clf.fit(scaled_data)
        anomalies = clf.predict(scaled_data)
        scores = clf.decision_function(scaled_data)
        return anomalies, scores
    
    elif method == 'dbscan':
        # DBSCAN clustering method
        # Reduce dimensionality first using PCA
        pca = PCA(n_components=min(10, scaled_data.shape[1]))
        reduced_data = pca.fit_transform(scaled_data)
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(reduced_data)
        
        # Treat noise points as anomalies
        anomalies = np.ones_like(clusters)
        anomalies[clusters == -1] = -1
        
        # No scores for DBSCAN
        return anomalies, None
    
    elif method == 'ensemble':
        # Ensemble approach combining multiple methods
        anomalies_iforest, scores_iforest = detect_anomalies(ndvi_series, method='iforest', contamination=contamination)
        anomalies_dbscan, _ = detect_anomalies(ndvi_series, method='dbscan')
        
        # Combine results (if either method flags as anomaly, consider it an anomaly)
        ensemble_anomalies = np.ones_like(anomalies_iforest)
        ensemble_anomalies[(anomalies_iforest == -1) | (anomalies_dbscan == -1)] = -1
        
        return ensemble_anomalies, scores_iforest
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from 'iforest', 'dbscan', or 'ensemble'.")

def visualize_anomalies(time_series, anomalies, timestamps=None, title="Anomaly Detection Results"):
    """
    Visualize the detected anomalies in a time series.
    
    Parameters:
    -----------
    time_series : numpy.ndarray
        1D array of time series values (e.g., mean NDVI for each date)
    anomalies : numpy.ndarray
        Array of labels where -1 indicates anomalies
    timestamps : list or numpy.ndarray, optional
        List of timestamps corresponding to the time series
    title : str
        Title for the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Create x-axis values
    if timestamps is None:
        x = np.arange(len(time_series))
    else:
        x = timestamps
    
    # Plot normal points
    normal_mask = anomalies == 1
    plt.scatter(x[normal_mask], time_series[normal_mask], c='blue', label='Normal', alpha=0.6)
    
    # Plot anomalies
    anomaly_mask = anomalies == -1
    plt.scatter(x[anomaly_mask], time_series[anomaly_mask], c='red', s=80, label='Anomaly', edgecolors='black')
    
    # Connect points with a line
    plt.plot(x, time_series, 'k-', alpha=0.3)
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels if timestamps are provided
    if timestamps is not None:
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    return plt

def spatial_anomaly_map(ndvi_image, anomalies, title="Spatial Distribution of Anomalies"):
    """
    Create a spatial map of anomalies for a single timestamp.
    
    Parameters:
    -----------
    ndvi_image : numpy.ndarray
        2D array representing NDVI values for a single timestamp
    anomalies : numpy.ndarray
        1D array of anomaly labels (-1 for anomalies, 1 for normal)
    title : str
        Title for the plot
    """
    # Reshape anomalies to match the spatial dimensions
    anomaly_map = anomalies.reshape(ndvi_image.shape)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot NDVI
    im0 = axes[0].imshow(ndvi_image, cmap='RdYlGn')
    axes[0].set_title('NDVI Values')
    axes[0].set_xlabel('X Coordinate')
    axes[0].set_ylabel('Y Coordinate')
    plt.colorbar(im0, ax=axes[0], label='NDVI')
    
    # Plot anomaly map
    im1 = axes[1].imshow(anomaly_map, cmap='coolwarm')
    axes[1].set_title('Anomaly Map')
    axes[1].set_xlabel('X Coordinate')
    axes[1].set_ylabel('Y Coordinate')
    plt.colorbar(im1, ax=axes[1], label='Normal (1) vs Anomaly (-1)')
    
    plt.suptitle(title)
    plt.tight_layout()
    return plt

def save_results(time_series, anomalies, scores=None, timestamps=None, output_dir='results'):
    """
    Save detection results to CSV file.
    
    Parameters:
    -----------
    time_series : numpy.ndarray
        1D array of time series values
    anomalies : numpy.ndarray
        Array of labels where -1 indicates anomalies
    scores : numpy.ndarray, optional
        Anomaly scores
    timestamps : list or numpy.ndarray, optional
        List of timestamps corresponding to the time series
    output_dir : str
        Directory to save results to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create timestamp if not provided
    if timestamps is None:
        timestamps = np.arange(len(time_series))
    
    # Create dataframe
    data = {'timestamp': timestamps, 'value': time_series, 'is_anomaly': anomalies == -1}
    
    if scores is not None:
        data['anomaly_score'] = scores
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f'anomaly_results_{timestamp}.csv')
    df.to_csv(filename, index=False)
    
    print(f"Results saved to {filename}")
    return filename

def process_time_series(data_cube, method='ensemble', visualize=True, save=True):
    """
    End-to-end processing of a time series of multispectral images.
    
    Parameters:
    -----------
    data_cube : numpy.ndarray
        4D array of shape (time, height, width, bands) representing a time series of multispectral images
    method : str
        Method to use for anomaly detection
    visualize : bool
        Whether to generate and display visualizations
    save : bool
        Whether to save results to disk
    
    Returns:
    --------
    results : dict
        Dictionary containing detection results
    """
    # Calculate vegetation indices
    ndvi_series = np.array([calculate_ndvi(data[np.newaxis, ...])[0] for data in data_cube])
    evi_series = np.array([calculate_evi(data[np.newaxis, ...])[0] for data in data_cube])
    
    # Calculate mean index value for each timestamp
    mean_ndvi = np.mean(ndvi_series, axis=(1, 2))
    
    # Detect anomalies
    anomalies, scores = detect_anomalies(ndvi_series, method=method)
    
    # Create timestamps (dummy dates for example)
    start_date = datetime(2023, 1, 1)
    timestamps = [start_date + pd.Timedelta(days=i*16) for i in range(len(mean_ndvi))]
    
    results = {
        'ndvi_series': ndvi_series,
        'evi_series': evi_series,
        'mean_ndvi': mean_ndvi,
        'anomalies': anomalies,
        'scores': scores,
        'timestamps': timestamps
    }
    
    # Visualize results
    if visualize:
        # Time series plot
        plt_ts = visualize_anomalies(mean_ndvi, anomalies, timestamps, 
                                   f"NDVI Time Series with Anomalies ({method.capitalize()} Method)")
        
        # Spatial anomaly map for the last timestamp
        plt_map = spatial_anomaly_map(ndvi_series[-1], anomalies[-1], 
                                    f"Spatial Distribution of Anomalies - {timestamps[-1].strftime('%Y-%m-%d')}")
        
        results['visualizations'] = {
            'time_series_plot': plt_ts,
            'spatial_map': plt_map
        }
    
    # Save results
    if save:
        results_file = save_results(mean_ndvi, anomalies, scores, timestamps)
        results['results_file'] = results_file
    
    return results

if __name__ == "__main__":
    # Example usage (with dummy data)
    print("Generating example data...")
    
    # Create a synthetic data cube (time, height, width, bands)
    # 10 timestamps, 50x50 pixels, 8 spectral bands
    np.random.seed(42)
    time_steps = 10
    height = width = 50
    bands = 8
    
    data_cube = np.random.rand(time_steps, height, width, bands) * 0.5 + 0.2
    
    # Introduce some anomalies
    # Create a healthy vegetation pattern (high NIR, low red)
    for t in range(time_steps):
        # Normal vegetation has high NIR and low red
        data_cube[t, :, :, 7] = 0.7 + 0.2 * np.random.rand(height, width)  # NIR (high)
        data_cube[t, :, :, 3] = 0.1 + 0.1 * np.random.rand(height, width)  # Red (low)
    
    # Add anomalies in specific regions and timestamps
    # Anomaly at t=5 in top-left corner
    data_cube[5, :20, :20, 7] = 0.2 + 0.1 * np.random.rand(20, 20)  # Lower NIR
    data_cube[5, :20, :20, 3] = 0.6 + 0.2 * np.random.rand(20, 20)  # Higher red
    
    # Anomaly at t=8 in bottom-right corner
    data_cube[8, 30:, 30:, 7] = 0.3 + 0.1 * np.random.rand(20, 20)  # Lower NIR
    data_cube[8, 30:, 30:, 3] = 0.5 + 0.2 * np.random.rand(20, 20)  # Higher red
    
    print("Processing data and detecting anomalies...")
    results = process_time_series(data_cube, method='ensemble', visualize=True, save=True)
    
    print("Done!")
    plt.show()
