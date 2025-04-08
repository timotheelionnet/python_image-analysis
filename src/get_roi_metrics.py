# Define a function that processes all the ROIs in one image:
# * it calculates the min background intensity in the image
# * if requested, it subtracts that min background value from the intensity in each ROI
# * it calculates the area of each ROI
# * it calculates the mean, median, STD, max and min intensity (minus background) for each ROI
# * it returns those metric

import numpy as np
import logging

# Import get_largest_background function from background_subtraction.py
from background_subtraction import get_largest_background

# The function takes two arguments: the image data and whether or not to do background subtraction (T/F)
def measureROIs(image_data, subtract_background=False):
    """
    Calculates metrics for each ROI in an image, optionally subtracting background.

    Parameters:
    - image_data (tuple): Contains (img_index, my_image, my_mask_n)
        - img_index (int): Index of the image being processed.
        - my_image (ndarray): The image data as a 3D numpy array (channels, height, width).
        - my_mask_n (ndarray): The corresponding mask data as a 2D numpy array.
    - subtract_background (bool): If True, subtracts the median background intensity from each ROI.

    Returns:
    - tuple: img_index (int), roi_metrics (dict) containing metrics for each ROI.
    """
    
    # Unpack the image data
    img_index, my_image, my_mask_n = image_data

    # Dictionary to store metrics for each ROI
    roi_metrics = {}
    my_rois = np.unique(my_mask_n)

    # Retrieve background intensities if subtraction is requested
    background_intensities = None
    if subtract_background:
        background_data = get_largest_background(my_mask_n, my_image, img_index)
        background_intensities = background_data['min_intensities']

    # Process each ROI (excluding background, labeled as '0')
    for roi in my_rois:
        if roi == 0:  # Skip the background ROI
            continue
        roi_metrics[roi] = {}
        area = np.sum(my_mask_n == roi)
        
        for channel_index in range(my_image.shape[0]):
            # Extract pixel values for the given ROI
            roi_pixels = my_image[channel_index, my_mask_n == roi]
            if roi_pixels.size == 0:
                logging.warning(f"ROI {roi} in Image {img_index}, Channel {channel_index} is empty.")
                continue

            # Perform background subtraction if requested
            # It will remove the min intensity of the largest background object in each image
            # from the intensity values of all the pixels in each ROI (non-zero ROIs)
            if subtract_background and background_intensities:
                roi_pixels = roi_pixels - background_intensities[channel_index]
                
            # Calculate metrics directly from the ROI pixels (with or without background subtraction)
            roi_metrics[roi][f'Channel {channel_index}'] = {
                'area': area,
                'avg': np.mean(roi_pixels),
                'median': np.median(roi_pixels),
                'std': np.std(roi_pixels),
                'maxima': np.max(roi_pixels),
                'minima': np.min(roi_pixels)
            }

    return img_index, roi_metrics