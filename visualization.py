import matplotlib.pyplot as plt
import numpy as np
import os

def plot_ndvi_series(ndvi_series, anomalies, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(ndvi_series.shape[1]):
        for j in range(ndvi_series.shape[2]):
            ts = ndvi_series[:, i, j]
            if np.any(ts):
                plt.plot(ts, label='NDVI')
                for k, a in enumerate(anomalies):
                    if a == -1:
                        plt.scatter(k, ts[k], color='red')
                plt.title(f"Pixel NDVI Series ({i}, {j})")
                plt.xlabel("Time Index")
                plt.ylabel("NDVI")
                plt.savefig(f"{output_dir}/ndvi_series_{i}_{j}.png")
                plt.clf()
                break
        break