import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import mapping
import numpy as np

def clip_raster_with_geodataframe(input_raster, gdf, output_raster, mask_value=None, 
                                  inverse=False, all_touched=False):
    """
    Masks a raster using the geometries in a GeoDataFrame.
    
    Parameters:
    -----------
    input_raster : str
        Path to the input raster file
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing the mask geometry(s)
    output_raster : str
        Path to save the masked raster
    mask_value : numeric, optional
        Value to assign to masked regions (default: src.nodata or 0)
    inverse : bool, optional
        If True, masks the areas OUTSIDE the geometry instead of inside
    all_touched : bool, optional
        If True, all pixels touched by the geometry are included in the mask
    """

    
    # Open the input raster
    with rasterio.open(input_raster) as src:
        # Get the CRS of the raster
        raster_crs = src.crs
        
        # Project the GeoDataFrame to match the raster CRS if necessary
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)
        
        # Convert the GeoDataFrame geometries to GeoJSON-like format for masking
        geometries = [mapping(geom) for geom in gdf.geometry]
        
        # Read the original data
        original_data = src.read()
        
        # Set default mask value
        if mask_value is None:
            mask_value = src.nodata if src.nodata is not None else 0
        
        # Create the mask
        if inverse:
            # Mask outside of the geometries
            mask_data, out_transform = mask(src, geometries, crop=False, 
                                          all_touched=all_touched, 
                                          invert=True)
          
            # Create a full raster of mask_values
            masked_data = np.full_like(original_data, mask_value)
            
            # Copy the unmasked data
            for i in range(original_data.shape[0]):
                masked_data[i, ~np.isnan(mask_data[i])] = original_data[i, ~np.isnan(mask_data[i])]
                
        else:
            # Mask inside of the geometries (standard approach)
            mask_data, out_transform = mask(src, geometries, crop=False, 
                                          all_touched=all_touched)
            
            # Copy the original data
            masked_data = original_data.copy()
            
            # Apply mask value to masked areas
            for i in range(original_data.shape[0]):
                masked_data[i, np.isnan(mask_data[i])] = mask_value
        
        # Update the metadata for the output raster
        out_meta = src.meta.copy()
        
        # Write the masked raster to disk
        with rasterio.open(output_raster, "w", **out_meta) as dest:
            dest.write(masked_data)
            
    print(f"Raster masked and saved to {output_raster}")
    return output_raster

def align_raster(input_raster, reference_raster, output_raster, resampling_method=Resampling.nearest):
    """
    Aligns a raster to match the CRS, resolution, and extent of a reference raster.
    
    Parameters:
    -----------
    input_raster : str
        Path to the input raster file
    reference_raster : str
        Path to the reference raster file
    output_raster : str
        Path to save the aligned raster
    resampling_method : Resampling, optional
        Method used for resampling during reprojection
    """
    # Open the input raster
    with rasterio.open(input_raster) as src:
        # Open the reference raster to get its metadata
        with rasterio.open(reference_raster) as reference:
            # Calculate the transformation parameters
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs, reference.crs,
                src.width, src.height,
                *src.bounds,
                dst_width=reference.width, dst_height=reference.height
            )
            
            # Update the metadata for the output raster
            dst_kwargs = src.meta.copy()
            dst_kwargs.update({
                'crs': reference.crs,
                'transform': dst_transform,
                'width': dst_width,
                'height': dst_height,
                'nodata': src.nodata if src.nodata is not None else 0
            })
            
            # Create the output raster
            with rasterio.open(output_raster, 'w', **dst_kwargs) as dst:
                # Reproject and save the data to the output raster
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=dst_transform,
                        dst_crs=reference.crs,
                        resampling=resampling_method
                    )
                
    print(f"Raster aligned and saved to {output_raster}")
    return output_raster
