# Define a function that processes all the ROIs in one image:
# * it calculates the mean background intensity in the image
# * it subtracts that background value from the intensity in each ROI
# * it calculates the area of each ROI
# * it calculates the mean, median, STD, max and min intensity (minus background) for each ROI

import numpy as np
import logging

def measureROIs(image_data):
    img_index, my_image, my_mask_n = image_data
    roi_metrics = {}
    my_rois = np.unique(my_mask_n)

    # Initialize a dictionary to store the background intensity for each channel
    background_intensities = {}

    # First, calculate the background intensity for each channel
    background_pixels = my_image[:, my_mask_n == 0]  # Assuming background is labeled as '0'
    if background_pixels.size > 0:
        for channel_index in range(my_image.shape[0]):
            background_intensities[channel_index] = np.mean(background_pixels[channel_index])
    else:
        logging.warning(f"No background pixels found for Image {img_index}. Using zero as background.")
        for channel_index in range(my_image.shape[0]):
            background_intensities[channel_index] = 0

    # Now process each ROI
    for roi in my_rois:
        if roi == 0:  # Skip the background ROI itself
            continue
        roi_metrics[roi] = {}
        area = np.sum(my_mask_n == roi)
        
        for channel_index in range(my_image.shape[0]):
            # Extract the values for the given ROI, if it exists (size not zero)
            roi_pixels = my_image[channel_index, my_mask_n == roi]
            if roi_pixels.size == 0:
                logging.warning(f"ROI {roi} in Image {img_index}, Channel {channel_index} is empty.")
                continue
                
            # Subtract mean background for that image from each ROI and calculate metrics
            bs_intensity = roi_pixels - background_intensities[channel_index]
            
            roi_metrics[roi][f'Channel {channel_index}'] = {
                'area': area,
                'avg': np.mean(bs_intensity),
                'median': np.median(bs_intensity),
                'std': np.std(bs_intensity),
                'maxima': np.max(bs_intensity),
                'minima': np.min(bs_intensity)
            }

    return img_index, roi_metrics

# Define an alternative version of the function that skips the background subtraction
# so we can compare the result of doing it to not doing it

def measureROIs_noBS(image_data):
    img_index, my_image, my_mask_n = image_data
    roi_metrics = {}
    my_rois = np.unique(my_mask_n)
    my_rois = my_rois[my_rois != 0]

    for roi in my_rois:
        area = np.sum(my_mask_n == roi)
        roi_metrics[roi] = {}

        for channel_index in range(my_image.shape[0]):
            roi_pixels = my_image[channel_index, my_mask_n == roi]
            if roi_pixels.size == 0:
                logging.warning(f"ROI {roi} in Image {img_index}, Channel {channel_index} is empty.")
                continue

            roi_metrics[roi][f'Channel {channel_index}'] = {
                'area': area,
                'avg': np.mean(roi_pixels),
                'median': np.median(roi_pixels),
                'std': np.std(roi_pixels),
                'maxima': np.max(roi_pixels),
                'minima': np.min(roi_pixels)
            }
            
    return img_index, roi_metrics