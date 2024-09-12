import os
import numpy as np
from PIL import Image


def load_masks_and_compute_average(mask_folder, background_percentile=1, rescale_percentile=99):
    """
    Load all masks from the specified folder, apply background removal and rescaling,
    compute the average pixel intensity for each mask, and then compute the overall average across all masks.

    Args:
        mask_folder (str): The folder where the mask images are stored.
        background_percentile (float): Percentile for background removal (default 1st percentile).
        rescale_percentile (float): Percentile for rescaling the image (default 99th percentile).

    Returns:
        float: The overall average pixel intensity across all masks.
    """
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.tif')]
    mask_averages = []

    print(f"Found {len(mask_files)} mask files.")

    # Loop through each mask file and compute the average pixel intensity
    for mask_file in mask_files:
        # Load the mask
        mask_path = os.path.join(mask_folder, mask_file)
        with Image.open(mask_path) as img:
            image_array = np.array(img, dtype=np.float32)

        # Background removal using the percentile strategy
        background_value = np.percentile(image_array, background_percentile)
        image_array = image_array - background_value  # Subtract the background
        image_array[image_array < 0] = 0  # Clip negative values to 0

        # Rescale the image based on the rescale percentile (e.g., 99th percentile)
        upper_value = np.percentile(image_array, rescale_percentile)
        if upper_value > 0:
            image_array = np.clip(image_array / upper_value, 0, 1)  # Clip values to ensure they are in [0, 1]

        # Compute the average pixel intensity for this mask after processing
        mask_avg = np.mean(image_array)
        mask_averages.append(mask_avg)

        # Print the average for this mask
        print(f"Average intensity for {mask_file} after processing: {mask_avg}")

    # Compute the overall average across all masks
    overall_average = np.mean(mask_averages)
    print(f"\nOverall average intensity across all masks: {overall_average}")

    return overall_average


