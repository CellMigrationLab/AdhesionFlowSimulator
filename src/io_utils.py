import os
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
from itertools import product
import time
import pandas as pd
import cv2
from phi.jax.flow import *
import json


# Function to generate filenames
def generate_filename(flow_speed, adhesion_strength, cell_density, run_id, mask_name):
    """
    Generate a filename for the simulation output, including the mask name or "uniform".

    Args:
        flow_speed (float): The flow speed in microns per second.
        adhesion_strength (float): The adhesion strength of the cells.
        cell_density (float): The cell density.
        run_id (int): The ID of the simulation run.
        mask_name (str): The name of the mask being used, or "uniform" if not using a mask.

    Returns:
        str: The generated filename.
    """
    return f'flow_{flow_speed}_as_{adhesion_strength}_cd_{cell_density}_run_{run_id}_mask_{mask_name}.csv'


# Function to save simulation parameters
def save_parameters(result_folder, flow_speed, adhesion_strength, cell_density, run_id, mask_name):
    """
    Save the simulation parameters to a CSV file, including the mask name.

    Args:
        result_folder (str): The folder where the parameters CSV file will be saved.
        flow_speed (float): The flow speed in microns per second.
        adhesion_strength (float): The adhesion strength of the cells.
        cell_density (float): The cell density.
        run_id (int): The ID of the simulation run.
        mask_name (str): The name of the mask being used, or "uniform" if not using a mask.
    """
    params_file = os.path.join(result_folder, 'simulation_parameters.csv')

    # Create a DataFrame with the simulation parameters
    params_df = pd.DataFrame({
        'Flow_Speed': [flow_speed],
        'Adhesion_Strength': [adhesion_strength],
        'Cell_Density': [cell_density],
        'Run_ID': [run_id],
        'Mask_Name': [mask_name],
        'Filename': [generate_filename(flow_speed, adhesion_strength, cell_density, run_id, mask_name)]
    })

    # Save the parameters to a CSV file, append if the file already exists
    if not os.path.exists(params_file):
        params_df.to_csv(params_file, index=False)
    else:
        params_df.to_csv(params_file, mode='a', header=False, index=False)


def create_video_from_positions(positions, video_path, field_size=(512, 512), cell_diameter_avg=10, total_simulation_time=10, frame_interval=1, fps=25):
    """Create a video from the saved positions using OpenCV."""

    # Check if positions is a file path or a numpy array
    if isinstance(positions, str):
        positions_df = pd.read_csv(positions, compression='gzip')
    else:
        positions_df = pd.DataFrame(positions, columns=['Track_ID', 'Frame_Num', 'X_Position', 'Y_Position', 'Status'])

    max_frame = int(positions_df['Frame_Num'].max()) + 1

    # Calculate the fps based on the total simulation time and number of frames if not provided
    if not fps:
        fps = max_frame / total_simulation_time

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (field_size[0], field_size[1]))

    # Use tqdm to show the progress bar
    for frame_num in tqdm(range(max_frame), desc="Creating Video"):
        # Create a blank white frame
        frame = np.ones((field_size[1], field_size[0], 3), dtype=np.uint8) * 255

        # Get the positions for this frame
        frame_positions = positions_df[positions_df['Frame_Num'] == frame_num]

        # Filter out inactive cells (status == 0) that are out of the field of view
        active_positions = frame_positions[(frame_positions['Status'] == 1) | (frame_positions['X_Position'] < field_size[0])]

        # Draw all circles for the current frame using OpenCV
        for _, row in active_positions.iterrows():
            x, y, status = int(row['X_Position']), int(row['Y_Position']), row['Status']
            color = (0, 0, 255) if status == 1 else (255, 0, 0)  # Red for attached, Blue for moving
            cv2.circle(frame, (x, y), int(cell_diameter_avg / 2), color, -1, lineType=cv2.LINE_AA)

        # Only write every nth frame to the video
        if frame_num % frame_interval == 0:
            video_writer.write(frame)

    video_writer.release()
    print(f"Video saved to {video_path}")


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


