#!/usr/bin/env python3

# This python script reads in the nuclear (or cell) masks generated for the indicated experiment
# counts all ROIs != 0 (background)
# and creates a nuclei-counts (or cell-counts) file in the parental folder 
# with the original file names and the number of cells
# Use --cells to count whole-cell masks instead.

import os
import re
import argparse
import numpy as np
import pandas as pd
from skimage import io

def process_images(experiment_folder: str, count_cells: bool = False):
    # Choose the mask subfolder based on the flag
    subdir = "whole-cell_masks" if count_cells else "nuclei_masks"
    folder_path = os.path.join(experiment_folder, "segmentation-masks", subdir)

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Mask folder not found: {folder_path}\n"
            f"(Check the experiment path and that '{subdir}' exists.)"
        )

    results = []

	# Process TIFF files deterministically (sorted)
    for file_name in sorted(os.listdir(folder_path)):
        # Only handle regular files with .tif/.tiff
        if not file_name.lower().endswith(('.tif', '.tiff')):
            continue
        file_path = os.path.join(folder_path, file_name)
        if not os.path.isfile(file_path):
            continue

        # Read mask
        image = io.imread(file_path)

        # Count non-zero unique labels
        unique_labels = np.unique(image)
        roi_labels = unique_labels[unique_labels != 0]
        roi_count = int(len(roi_labels))

        # Build display filename:
        # 1) drop leading "C{digit}-" if present (e.g., "C1-")
        # 2) change "_MIP_cp_masks.tif" -> ".ome.tif"
        new_filename = file_name
        new_filename = re.sub(r"^C\d-", "", new_filename)
        new_filename = new_filename.replace("_MIP_cp_masks.tif", ".ome.tif")

        results.append({"filename": new_filename, "total_cells": roi_count})

    # Create DataFrame and save
    counts_df = pd.DataFrame(results)

    out_name = "cell-counts.csv" if count_cells else "nuclei-counts.csv"
    csv_path = os.path.join(experiment_folder, out_name)
    counts_df.to_csv(csv_path, index=False)
    print(f"Saved counts to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count ROI labels in microscopy mask TIFFs and save as CSV. "
                    "By default counts nuclei masks; pass --cells to count whole-cell masks."
    )
    parser.add_argument(
        "experiment_folder",
        type=str,
        help="Path to the parent folder of the experiment"
    )
    parser.add_argument(
        "--cells",
        action="store_true",
        help="Count whole-cell masks instead of nuclei masks"
    )
    args = parser.parse_args()
    process_images(args.experiment_folder, count_cells=args.cells)
    