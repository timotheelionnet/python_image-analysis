#!/usr/bin/env python3

# This python script reads in the nuclear masks generated for the indicated experiment
# counts all ROIs != 0 (background)
# and creates a nuclei-counts file in the parental folder with the original file names and the number of cells

import os
import numpy as np
import pandas as pd
from skimage import io
import argparse

def process_images(experiment_folder):
	# Construct the full path to the mask TIFF files subdirectory 
	folder_path = os.path.join(experiment_folder, "segmentation-masks", "nuclei_masks")

	# Prepare a list to store the results
	results = []

	# Loop over each file in the folder
	for file_name in os.listdir(folder_path):
	    # Process only TIFF files (supports .tif and .tiff extensions)
	    if file_name.lower().endswith(('.tif', '.tiff')):
	        file_path = os.path.join(folder_path, file_name)
	        # Read the TIFF file
	        image = io.imread(file_path)
	        # Find the unique ROI labels and exclude the background (0)
	        unique_labels = np.unique(image)
	        roi_labels = unique_labels[unique_labels != 0]
	        # Count the number of ROIs
	        roi_count = len(roi_labels)

	        # Modify the filename:
	        # Remove the "C1-" prefix and replace "_cp_masks.tif" with ".tif"
	        new_filename = file_name
	        if new_filename.startswith("C1-"):
	            new_filename = new_filename[3:]
	        new_filename = new_filename.replace("_MIP_cp_masks.tif", ".ome.tif")
	        
	        # Append the result with new column names
	        results.append({"filename": new_filename, "total_cells": roi_count})

	# Create a DataFrame from the results list
	cell_counts = pd.DataFrame(results)

	# Define the output CSV file path in the parent_folder
	csv_path = os.path.join(experiment_folder, "nuclei-counts.csv")
	# Save the DataFrame as a CSV file (without the index column)
	cell_counts.to_csv(csv_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process microscopy TIFF files to count nuclei ROIs and save the result as a CSV."
    )
    parser.add_argument(
        "experiment_folder",
        type=str,
        help="Path to the parent folder of the experiment"
    )
    args = parser.parse_args()
    process_images(args.experiment_folder)
