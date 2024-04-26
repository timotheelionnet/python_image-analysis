# outliers_removal.py

# This script makes a function that filters out ROIs in a dataset that fall outside a given StdDev cutoff

# The function assumes the dataset is a dictionary with a nested/hierarchical structure:
#   Top level has a series of identifier-image file key-value pairs
#   Second level has ROIs contained within each image file, the area of which is being filtered

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def outliersOut(df, sd_filter = 2, transformation = 'log'):
    # Apply the transformation (log or sqrt)
    if transformation == 'log':
        df['tx_area'] = np.log(df['area'] + 1)
    elif transformation == 'sqrt':
        df['tx_area'] = np.sqrt(df['area'])
    else:
        raise ValueError("Transformation should be either 'log' or 'sqrt'.")


    # Fit a normal distribution
    mean, std_dev = norm.fit(df['tx_area'])

    # Calculate the threshold for filtering
    lower_threshold = mean - (sd_filter * std_dev)
    upper_threshold = mean + (sd_filter * std_dev)

    # Filter out outliers
    filtered_df = df[(df['tx_area'] >= lower_threshold) & (df['tx_area'] <= upper_threshold)]

    return filtered_df, (mean, std_dev, lower_threshold, upper_threshold)
