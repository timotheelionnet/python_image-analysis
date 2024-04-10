# outliers_removal.py

# This script makes a function that filters out ROIs in a dataset that fall outside a given StdDev cutoff

# The function assumes the dataset is a dictionary with a nested/hierarchical structure:
#   Top level has a series of identifier-image file key-value pairs
#   Second level has ROIs contained within each image file, the area of which is being filtered

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def outliersOut(all_roi_metrics, sd_filter = 2, transformation = 'log'):
    # Collect all ROI areas
    all_areas = []
    for image_metrics in all_roi_metrics.values():
        for roi_key, roi_metrics in image_metrics.items():
            first_channel_key = next(iter(roi_metrics))
            area = roi_metrics[first_channel_key]['area']
            all_areas.append(area)

    # Apply the transformation (log or sqrt)
    if transformation == 'log':
        tx_areas = np.log(np.array(all_areas) + 1)  # Added 1 to avoid log(0)
    elif transformation == 'sqrt':
        tx_areas = np.sqrt(all_areas)
    else:
        raise ValueError("Transformation should be either 'log' or 'sqrt'.")

    # Fit a normal distribution
    mean, std_dev = norm.fit(tx_areas)

    # Calculate the threshold for filtering
    lower_threshold = mean - (sd_filter * std_dev)
    upper_threshold = mean + (sd_filter * std_dev)

    # Filter out ROIs
    filtered_roi_metrics = {}
    for image_key, image_metrics in all_roi_metrics.items():
        filtered_image_metrics = {}
        for roi_key, roi_metrics in image_metrics.items():
            for channel_key, channel_metrics in roi_metrics.items():
                area = channel_metrics['area']
                tx_area = np.log(area + 1) if transformation == 'log' else np.sqrt(area)
                if lower_threshold <= tx_area <= upper_threshold:
                    filtered_image_metrics[roi_key] = roi_metrics
        if filtered_image_metrics:
            filtered_roi_metrics[image_key] = filtered_image_metrics

    # Collect areas of filtered ROIs for the histogram
    filtered_areas = []
    for image_metrics in filtered_roi_metrics.values():
        for roi_metrics in image_metrics.values():
            for channel_metrics in roi_metrics.values():
                filtered_areas.append(channel_metrics['area'])

    # Apply the transformation (log or sqrt) to the filtered areas
    if transformation == 'log':
        tx_filtered_areas = np.log(np.array(filtered_areas) + 1)
    elif transformation == 'sqrt':
        tx_filtered_areas = np.sqrt(filtered_areas)
    else:
        raise ValueError("Transformation should be either 'log' or 'sqrt'.")

    return filtered_roi_metrics, (mean, std_dev, lower_threshold, upper_threshold)
