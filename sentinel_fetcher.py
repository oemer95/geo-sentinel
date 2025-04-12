# sentinel_fetcher.py
import numpy as np
import datetime
from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions, SentinelHubDownloadClient, SentinelHubCatalog, filter_times

def load_config():
    import yaml
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

def create_bbox(aoi):
    return BBox(bbox=aoi, crs=CRS.WGS84)

def get_time_series_data(aoi, time_range, bands, resolution, config):
    bbox = create_bbox(aoi)
    size = bbox_to_dimensions(bbox, resolution=resolution)
    
    evalscript = f"""
    //VERSION=3
    function setup() {{
        return {{
            input: [{', '.join([f'"{b}"' for b in bands])}],
            output: {{ bands: {len(bands)}, sampleType: "FLOAT32" }}
        }};
    }}
    function evaluatePixel(sample) {{
        return [{', '.join([f'sample.{b}' for b in bands])}];
    }}
    """
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[SentinelHubRequest.input_data(data_collection=DataCollection.SENTINEL2_L2A)],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=size,
        config=config,
        time_interval=time_range,
    )

    return request.get_data()

def setup_sh_config(client_id, client_secret):
    sh_config = SHConfig()
    sh_config.sh_client_id = client_id
    sh_config.sh_client_secret = client_secret
    sh_config.save()
    return sh_config