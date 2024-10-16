# A routine to do the following over a series of nuclear masks and image pairs:
# 1. Find all the background object(s) in the nuclear masks (ROI == 0) 
# 2. Calculate the area of each background object
# 3. Select the largest background object for each mask (true background)
# 4. Calculate the average and median intensity on each channel for that background object
# 5. Return area, intensity metrics and the image index

import numpy as np
from scipy.ndimage import label, find_objects

def get_largest_background(mask, image, image_index):
    """
    Analyzes the largest background region in a given nuclear mask.
    
    Parameters:
    - mask: A 2D numpy array where 0 indicates background.
    - image: A 3D numpy array (channels, height, width) corresponding to the mask.
    
    Returns:
    - A dictionary containing the area and average intensities of the largest background region for each channel.
    """
    # Identify background regions and label them
    background = (mask == 0)
    labeled_bg, num_features = label(background)
    
    # Ensure background was found
    if num_features == 0:
        return {
            'area': 0, 
            'average_intensities': [0] * image.shape[0], 
            'median_intensities': [0] * image.shape[0], 
            'image_index' : image_index
        }
    
    # Find the area of each background object
    object_slices = find_objects(labeled_bg)
    areas = [np.sum(labeled_bg[obj] > 0) for obj in object_slices]
    
    # Determine the largest background object
    largest_index = np.argmax(areas)
    largest_slice = object_slices[largest_index]
    
    # Calculate the average intensity for each channel in the largest background area
    avg_intensities = []
    median_intensities = []
    for channel in range(image.shape[0]):
        channel_data = image[channel]
        largest_region = channel_data[largest_slice]
        masked_region = largest_region[labeled_bg[largest_slice] == largest_index + 1]
        avg_intensity = np.mean(masked_region)
        median_intensity = np.median(masked_region)
        avg_intensities.append(avg_intensity)
        median_intensities.append(median_intensity)
    
    return {
        'area': areas[largest_index],
        'average_intensities': avg_intensities,
        'median_intensities': median_intensities,
        'image_index': image_index
    }

# Example usage, assuming you have a `mask` and its corresponding `image`
# mask = all_images['mask_n'][some_index]
# image = all_images['image'][some_index]
# result = get_largest_background(mask, image)
# print(result)