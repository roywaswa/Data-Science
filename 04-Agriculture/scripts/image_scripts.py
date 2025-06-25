import rasterio

def read_image(image_path):
    """
    Reads an image from the specified path using rasterio.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The image data as a NumPy array.
    """
    with rasterio.open(image_path) as src:
        image_data = src.read()
    return image_data


