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


# Define function to check dimensions of images and masks
def img_sizr(image_types, file_indices):
    for f in file_indices:
        for image_type in image_types:
            try:
                image_shape = all_images[image_type][f].shape
                logging.info(f"File {f} ({image_type}): {image_shape}")
            except IndexError:
                logging.warning(f"File index {f} for {image_type} is out of range.")
            except KeyError:
                logging.error(f"Image type {image_type} does not exist.")
