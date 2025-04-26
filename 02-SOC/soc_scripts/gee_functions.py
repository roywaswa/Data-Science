
# -----------------------------------------------------------------
#              Sentinel1SARIndices Class
# -----------------------------------------------------------------

class Sentinel1SARIndices:

    """
    A class to compute various SAR indices for Sentinel-1 images using Google Earth Engine.
    This class provides methods to compute the following indices:  
    - VH/VV Ratio Index (VVVHR)
    - Radar Vegetation Index (RVI)
    - Normalized Difference Index (NDI)
    - Cross-Polarization Ratio (CPR)
    
    Each method computes the respective index and adds it as a new band to the image.
    The computed indices can be used for further analysis or visualization.
    The class is designed to be used with Google Earth Engine (GEE) Python API.
    """
    def __init__(self, image):
        """
        Initialize the Sentinel1SARIndices class with an Earth Engine image.

        Args:
            image (ee.Image): The Sentinel-1 image to process.
        """
        self.image = image

    def compute_vvvhr(self):
        """Computes VH/VV Ratio Index for Sentinel-1 Image."""
        vh = self.image.select('VH')
        vv = self.image.select('VV')
        vvvhr = vh.divide(vv).rename('VVVHR')
        self.image = self.image.addBands(vvvhr)
        return self

    def compute_rvi(self):
        """Computes Radar Vegetation Index (RVI) for Sentinel-1 Image."""
        vh = self.image.select('VH')
        vv = self.image.select('VV')
        rvi = (vh.multiply(4)).divide(vv.add(vh)).rename('RVI')
        self.image = self.image.addBands(rvi)
        return self

    def compute_ndi(self):
        """Computes Normalized Difference Index (NDI) for Sentinel-1 Image."""
        vh = self.image.select('VH')
        vv = self.image.select('VV')
        ndi = vv.subtract(vh).divide(vv.add(vh)).rename('NDI')
        self.image = self.image.addBands(ndi)
        return self

    def compute_cpr(self):
        """Computes Cross-Polarization Ratio (CPR) for Sentinel-1 Image."""
        vh = self.image.select('VH')
        vv = self.image.select('VV')
        cpr = vh.divide(vv).rename('CPR')
        self.image = self.image.addBands(cpr)
        return self

    def apply_all_indices(self):
        """Applies all Sentinel-1 SAR indices to the image."""
        return (
            self.compute_vvvhr()
                .compute_rvi()
                .compute_ndi()
                .compute_cpr()
        ).image
        

# Example usage:
# Initialize the class with a Sentinel-1 image
# sentinel1_image = ee.Image('COPERNICUS/S1_GRD/20200101T000000_20200101T000000_S1A_IW_GRDH_1SDV_001')
# sar_indices = Sentinel1SARIndices(sentinel1_image)
# Compute all indices
# processed_image = sar_indices.apply_all_indices().image


# -----------------------------------------------------------------
#              Sentinel2Indices Class
# -----------------------------------------------------------------
class Sentinel2Indices:
    """A class to compute various spectral indices from Sentinel-2 imagery."""
    
    def __init__(self, image):
        """
        Initialize with a Sentinel-2 Earth Engine image.

        Args:
            image (ee.Image): The Sentinel-2 image to process.
        """
        self.image = image

    def compute_ndvi(self):
        """Computes NDVI (Normalized Difference Vegetation Index)."""
        ndvi = self.image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        self.image = self.image.addBands(ndvi)
        return self

    def compute_ndsi(self):
        """Computes NDSI (Normalized Difference Soil Index)."""
        ndsi = self.image.normalizedDifference(['B11', 'B4']).rename('NDSI')
        self.image = self.image.addBands(ndsi)
        return self

    def compute_savi(self, L=0.5):
        """
        Computes SAVI (Soil Adjusted Vegetation Index).
        
        Args:
            L (float): Soil brightness correction factor. Default is 0.5.
        """
        nir = self.image.select('B8')
        red = self.image.select('B4')
        savi = ((nir.subtract(red)).multiply(1 + L)).divide(nir.add(red).add(L)).rename('SAVI')
        self.image = self.image.addBands(savi)
        return self

    def compute_bsi(self):
        """Computes BSI (Bare Soil Index)."""
        swir = self.image.select('B11')
        red = self.image.select('B4')
        nir = self.image.select('B8')
        blue = self.image.select('B2')
        bsi = ((swir.add(red)).subtract(nir.add(blue))).divide((swir.add(red)).add(nir.add(blue))).rename('BSI')
        self.image = self.image.addBands(bsi)
        return self

    def compute_nbr(self):
        """Computes NBR (Normalized Burn Ratio)."""
        nbr = self.image.normalizedDifference(['B8', 'B12']).rename('NBR')
        self.image = self.image.addBands(nbr)
        return self

    def apply_all_indices(self):
        """
        Applies all Sentinel-2 indices to the image.
        
        Returns:
            ee.Image: The input image with all spectral indices added as bands.
        """
        return (self
                .compute_ndvi()
                .compute_ndsi()
                .compute_savi()
                .compute_bsi()
                .compute_nbr()
                .image)

# Function to download Sentinel-2 ee.image.Image per band, then stack them using rasterio
import geemap as gm
import os
import rasterio
def download_sentinel_bands(image, output_dir,filename, region=None):
    """
    Downloads specified bands from a Sentinel-2 image and stacks them into a single GeoTIFF file.

    Args:
        image (ee.Image): The Sentinel-2 image to process.
        output_dir (str): Directory to save the output GeoTIFF file.

    Returns:
        str: Path to the saved GeoTIFF file.
    """
    # download each band separately
    path_to_image = os.path.join(output_dir, filename)
    gm.ee_export_image(image, filename=path_to_image, scale=20, region=region, file_per_band=True)
    
    # get the list of downloaded bands
    band_files = [f for f in os.listdir(output_dir) if f.endswith('.tif')]
    band_files.sort()  # Sort the files to ensure correct stacking order
    
    # stack the bands using rasterio
    stacked_file = os.path.join(output_dir, 'stacked_image.tif')
    with rasterio.open(os.path.join(output_dir, band_files[0])) as src:
        meta = src.meta.copy()
        meta.update(count=len(band_files))
        with rasterio.open(stacked_file, 'w', **meta) as dst:
            for i, band_file in enumerate(band_files, start=1):
                with rasterio.open(os.path.join(output_dir, band_file)) as src_band:
                    dst.write(src_band.read(1), i)

