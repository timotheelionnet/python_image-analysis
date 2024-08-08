# This script defines two simple functions:
# One to load images and mask files for joint processing
# One to determine image dimensions

import os
import logging
import tifffile as tiff

# Define function to load images and masks
def img_loadr(file_path):
    try:
        image = tiff.imread(file_path)
        # logging.info(f"Loaded image {os.path.basename(file_path)} successfully.")
        return image, os.path.basename(file_path)
    except Exception as e:
        error_msg = f"Error loading image {file_path}: {str(e)}"
        logging.error(error_msg)
        errors.append(error_msg)
        return None, file_path