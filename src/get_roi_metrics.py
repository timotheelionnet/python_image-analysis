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
def measureROIs(
    image_data, 
    subtract_background: bool = False, 
    background_stat: str = "min",    # Values allowed are "min", "median", "mean". Defaluts to 'min'
    clip_after_subtraction: bool = False,
    remove_edges: bool = False,
    edge_margin: int = 100,
    edge_method: str = "centroid"  # "centroid" (default) or "bbox"
    ):
    """
    Calculates metrics for each ROI in an image, with optional background subtraction 
    (using min/median/mean from the largest background region)
    and optional edge filtering.

    Parameters:
    - image_data (tuple): Contains (img_index, my_image, my_mask_n)
        - img_index (int): Index of the image being processed.
        - my_image (ndarray): The image data as a 3D numpy array (channels, height, width).
        - my_mask_n (ndarray): The corresponding mask data as a 2D numpy array.
    - subtract_background (bool): If True, subtracts the min background intensity from each ROI.
    - remove_edges (bool): if True, skip ROIs near image edges
    - edge_margin (int): margin (in pixels) used by the edge filter
    - edge_method (str): "centroid" (distance from edges using ROI centroid)
                         or "bbox" (distance using ROI bounding box)

    Returns:
    - tuple: (img_index, roi_metrics) where roi_metrics is a dict keyed by ROI label.
    """
    
    # Unpack the image data
    img_index, my_image, my_mask_n = image_data

    # Get image shape
    H, W = my_mask_n.shape

    # Dictionary to store metrics for each ROI
    roi_metrics = {}
    my_rois = np.unique(my_mask_n)

    # Helper to decide if an ROI is too close to the image border
    def _is_near_edge(coords):
        # coords: Nx2 array of (row, col) = (y, x) pixel indices
        if coords.size == 0:
            return False  # let empty handling happen downstream
        ys, xs = coords[:, 0], coords[:, 1]

        if edge_method == "bbox":
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            left   = xmin
            right  = (W - 1) - xmax
            top    = ymin
            bottom = (H - 1) - ymax
        else:  # "centroid" (default)
            cy, cx = ys.mean(), xs.mean()
            left   = cx
            right  = (W - 1) - cx
            top    = cy
            bottom = (H - 1) - cy

        return (left < edge_margin) or (right < edge_margin) or (top < edge_margin) or (bottom < edge_margin)

    # Warn if the margin is larger than half the image — likely to exclude most ROIs
    if remove_edges and (edge_margin*2 >= min(H, W)):
        logging.warning(
            f"Image {img_index}: edge_margin={edge_margin} is large relative to image size ({H}x{W}); "
            "most or all ROIs may be excluded."
        )

    # Retrieve background intensities if subtraction is requested
    background_values = None
    if subtract_background:
        bg = get_largest_background(my_mask_n, my_image, img_index)
        key_map = {
            "min": "min_intensities",
            "median": "median_intensities",
            "mean": "average_intensities"
        }
        background_stat = (background_stat or "min").lower()
        if background_stat not in key_map:
            logging.warning(f"Unknown background_stat='{background_stat}', defaulting to 'min'.")
            background_stat = "min"
        stat_key = key_map[background_stat]
        if stat_key not in bg:
            # Extremely defensive; your helper already provides these.
            logging.warning(f"Background key '{stat_key}' missing; defaulting to min_intensities.")
            stat_key = "min_intensities"
        background_values = np.asarray(bg[stat_key], dtype=float)

    # Main loop. Process each ROI (excluding background, labeled as '0')
    for roi in my_rois:
        if roi == 0:  # Skip the background ROI
            continue

        # Get coordinates for this ROI once (used by edge filter and area)
        coords = np.argwhere(my_mask_n == roi)  # (N, 2) as (row=y, col=x)

        if remove_edges and _is_near_edge(coords):
            logging.debug(f"Skipping ROI {roi} in Image {img_index}: near edge (method={edge_method}, margin={edge_margin}).")
            continue

        area = coords.shape[0]
        if area == 0:
            logging.warning(f"ROI {roi} in Image {img_index} is empty.")
            continue

        ys, xs = coords[:, 0], coords[:, 1]
        roi_metrics[roi] = {}

        for channel_index in range(my_image.shape[0]):
            # Extract pixel values efficiently using coords
            roi_pixels = my_image[channel_index, ys, xs].astype(float)

            if roi_pixels.size == 0:
                logging.warning(f"ROI {roi} in Image {img_index}, Channel {channel_index} is empty.")
                continue


            # Optional: background subtraction
            # It will remove the min intensity of the largest background object in each image
            # from the intensity values of all the pixels in each ROI (non-zero ROIs)
            if subtract_background and background_values is not None:
                roi_pixels = roi_pixels - background_values[channel_index]
                if clip_after_subtraction:
                    roi_pixels = np.clip(roi_pixels, 0, None)
                
            # Calculate metrics directly from the ROI pixels (with or without background subtraction)
            roi_metrics[roi][f'Channel {channel_index}'] = {
                'area': area,
                'avg': float(np.mean(roi_pixels)),
                'median': float(np.median(roi_pixels)),
                'std': float(np.std(roi_pixels)),
                'maxima': float(np.max(roi_pixels)),
                'minima': float(np.min(roi_pixels)),
            }

    return img_index, roi_metrics