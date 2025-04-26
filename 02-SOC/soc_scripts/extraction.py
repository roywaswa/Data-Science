# This are all the scripts used to obtain data from the various sources to be used in the analysis
# and the model.

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import Point
import geopandas as gpd
import os


# -----------------------------------------------------------------
#                  CREATE POINTS FROM RASTER
# -----------------------------------------------------------------

# Create a function to extract the pixel centroid points from a raster
# This function creates points at the center of each pixel in a raster and returns a GeoDataFrame.
# It can also save the points to a shapefile if an output path is provided.
def create_pixel_centroid_points(raster_path, output_points_path=None, sample_percentage=100):
    """
    Create points at the center of each pixel in a raster.
    
    Parameters:
    -----------
    raster_path : str
        Path to the input raster file
    output_points_path : str, optional
        Path to save the output points shapefile
    sample_percentage : float, optional
        Percentage of pixels to sample (1-100), useful for large rasters
        
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame containing point geometries at pixel centers
    """
    # Read the raster
    with rasterio.open(raster_path) as src:
        # Get raster metadata
        transform = src.transform
        crs = src.crs
        
        # Read the raster data
        data = src.read(1)
        
        # Create empty lists to store points and their row/col indices
        points = []
        rows = []
        cols = []
        
        # Get dimensions
        height, width = data.shape
        
        # Determine which pixels to sample
        if sample_percentage < 100:
            # Create a mask for random sampling
            sample_size = int((height * width) * (sample_percentage / 100))
            flat_indices = np.random.choice(height * width, size=sample_size, replace=False)
            sampled_rows, sampled_cols = np.unravel_index(flat_indices, (height, width))
            
            # Create iterator for sampled pixels
            pixels = zip(sampled_rows, sampled_cols)
        else:
            # Create iterator for all pixels
            pixels = [(r, c) for r in range(height) for c in range(width)]
        
        # Create points at the center of each pixel
        for row, col in pixels:
            if not np.isnan(data[row, col]):  # Skip NoData pixels
                # Get x, y coordinates of the pixel center
                x, y = rasterio.transform.xy(transform, row, col, offset='center')
                
                # Create point and add to list
                points.append(Point(x, y))
                rows.append(row)
                cols.append(col)
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame({
            'geometry': points,
            'row': rows,
            'col': cols
        }, crs=crs)
        
        # Save to file if output path is provided
        if output_points_path:
            gdf.to_file(output_points_path)
        
        return gdf

# # Example usage
# raster_path = 'path/to/your/raster.tif'
# output_points_path = 'path/to/output/points.shp'
# points_gdf = create_pixel_centroid_points(raster_path, output_points_path, sample_percentage=10)


# -----------------------------------------------------------------
#                  SAMPLE RASTERS AT POINTS
# -----------------------------------------------------------------

# This function samples values from multiple rasters at specified point locations.
# It takes a GeoDataFrame with point geometries and a dictionary of raster paths, and returns a new GeoDataFrame with the sampled values.
def sample_rasters_at_points(point_gdf, raster_paths):
    """
    Sample values from multiple rasters at point locations.
    
    Parameters:
    -----------
    point_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing point geometries
    raster_paths : dict
        Dictionary with variable names as keys and raster paths as values
        
    Returns:
    --------
    geopandas.GeoDataFrame
        Original GeoDataFrame with additional columns for each sampled raster
    """
    # Make a copy of the input GeoDataFrame
    result_gdf = point_gdf.copy()
    
    # Ensure points are in the same CRS as the rasters
    # Convert to the CRS of the first raster
    first_raster_crs = rasterio.open(list(raster_paths.values())[0]).crs
    if point_gdf.crs != first_raster_crs:
        point_gdf = point_gdf.to_crs(first_raster_crs)
    
    # For each raster, extract values at the point locations
    for var_name, raster_path in raster_paths.items():
        print(f"Sampling {var_name} from {raster_path}")
        
        # Read the raster
        with rasterio.open(raster_path) as src:
            # Sample values at point locations
            values = []
            for point in point_gdf.geometry:
                # Get row, col for this point
                row, col = rowcol(src.transform, point.x, point.y)
                
                # Check if point is within raster bounds
                if 0 <= row < src.height and 0 <= col < src.width:
                    value = src.read(1)[row, col]
                    if src.nodata is not None and value == src.nodata:
                        values.append(np.nan)
                    else:
                        values.append(value)
                else:
                    values.append(np.nan)
            
            # Add values as a new column
            result_gdf[var_name] = values
    
    return result_gdf

# # Example usage
# point_gdf = gpd.read_file('path/to/your/points.shp')
# raster_paths = {
#     'variable1': 'path/to/raster1.tif',
#     'variable2': 'path/to/raster2.tif',
#     # Add more variables as needed
# }
# sampled_gdf = sample_rasters_at_points(point_gdf, raster_paths)


# -----------------------------------------------------------------
#                  CONVERT POINTS TO RASTER
# -----------------------------------------------------------------

# This function converts point data with values to a raster format.
# It uses a template raster to determine the extent and resolution of the output raster.
def points_to_raster(points_gdf, template_raster_path, output_raster_path, value_column='soc_predicted'):
    """
    Convert points with values to a raster using a template raster for extent and resolution.
    
    Parameters:
    -----------
    points_gdf : geopandas.GeoDataFrame
        GeoDataFrame with point geometries and values to rasterize
    template_raster_path : str
        Path to the template raster file to use for extent and resolution
    output_raster_path : str
        Path to save the output raster
    value_column : str, optional
        Name of the column containing values to rasterize
        
    Returns:
    --------
    str
        Path to the created raster file
    """
    # Read template raster to get metadata
    with rasterio.open(template_raster_path) as src:
        # Get metadata from template
        out_meta = src.meta.copy()
        out_meta.update(dtype='float32', nodata=-9999)
        
        # Create empty array with same dimensions as template
        out_array = np.full((src.height, src.width), out_meta['nodata'], dtype='float32')
        
        # Iterate through points and place values in the array
        for idx, point in points_gdf.iterrows():
            row, col = rowcol(src.transform, point.geometry.x, point.geometry.y)
            
            # Check if point is within array bounds
            if 0 <= row < src.height and 0 <= col < src.width:
                if not pd.isna(point[value_column]):
                    out_array[row, col] = point[value_column]
        
        # Write the output raster
        with rasterio.open(output_raster_path, 'w', **out_meta) as dst:
            dst.write(out_array, 1)
    
    return output_raster_path

# # Example usage
# points_gdf = gpd.read_file('path/to/your/points.shp')
# template_raster_path = 'path/to/template/raster.tif'
# output_raster_path = 'path/to/output/raster.tif'
# points_to_raster(points_gdf, template_raster_path, output_raster_path, value_column='soc_predicted')

import os
import rasterio
import glob

def stack_rasters(directory, output_filename):
    """
    Stack all raster files in a directory into one multi-band raster file.
    Files are expected to have format: rastername.{bandname}.tif
    
    Args:
        directory (str): Path to directory containing raster files
        output_filename (str): Name of output stacked file (will be saved in the same directory)
    
    Returns:
        str: Path to the stacked output file
    """
    # Ensure output has .tif extension
    if not output_filename.endswith('.tif'):
        output_filename = f"{output_filename}.tif"
    
    # Normalize directory path to ensure proper path handling
    directory = os.path.normpath(directory)
    
    # Get all tif files in the directory
    raster_files = glob.glob(os.path.join(directory, '*.tif'))
    
    if not raster_files:
        raise ValueError(f"No raster files found in {directory}")
    
    # Sort files for consistent ordering
    raster_files.sort()
    
    print(f"Found {len(raster_files)} raster files to stack")
    
    # Get metadata from first raster
    with rasterio.open(raster_files[0]) as src:
        meta = src.meta.copy()
    
    # Update metadata for the stacked file
    meta.update(count=len(raster_files))
    
    # Full path for output file - make sure we're using proper path joining
    output_path = os.path.join(directory, output_filename)
    print(f"Will create stacked raster at: {output_path}")
    
    # Stack all rasters
    with rasterio.open(output_path, 'w', **meta) as dst:
        for i, raster_path in enumerate(raster_files, start=1):
            with rasterio.open(raster_path) as src:
                band_data = src.read(1)
                dst.write(band_data, i)
                
                # Extract band name from filename (middle part between dots)
                filename = os.path.basename(raster_path)
                parts = filename.split('.')
                if len(parts) >= 3:
                    # For filename format like "rastername.B8.tif"
                    band_name = parts[-2]  # Get the second-to-last part (before .tif)
                else:
                    # Fallback if the filename doesn't match expected format
                    band_name = f"Band_{i}"
                    
                print(f"  Band {i}: {band_name} from {filename}")
                dst.update_tags(i, name=band_name)
    
    print(f"Stacked raster created at: {output_path}")
    return output_path
