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


def load_receptor_map(image_path, field_size=(512, 512), background_percentile=1, rescale_percentile=99):
    """
    Load the receptor map from a TIFF image, remove background using a percentile, rescale using another percentile,
    and ensure it matches the field size.

    Args:
        image_path (str): Path to the receptor map image (TIFF format).
        field_size (tuple): Expected size of the output image (default is 512x512).
        background_percentile (float): The percentile for background subtraction (default is 1st percentile).
        rescale_percentile (float): The percentile for rescaling the image to [0, 1] (default is 99th percentile).

    Returns:
        np.ndarray: The processed image array, rescaled to 0-1 and with the background removed.
    """
    # Load the image
    original_image = Image.open(image_path)

    # Convert to NumPy array
    image_array = np.array(original_image, dtype=np.float32)

    # Resize the image to match the field size if needed
    if image_array.shape != field_size:
        image_array = np.array(original_image.resize(field_size, Image.BILINEAR), dtype=np.float32)

    # Background removal using the percentile strategy
    background_value = np.percentile(image_array, background_percentile)
    image_array = image_array - background_value  # Subtract the background
    image_array[image_array < 0] = 0  # Clip negative values to 0

    # Rescale the image based on the rescale percentile (e.g., 99th percentile)
    upper_value = np.percentile(image_array, rescale_percentile)
    if upper_value > 0:
        image_array = np.clip(image_array / upper_value, 0, 1)  # Clip values to ensure they are in [0, 1]

    return image_array

# Function to plot heatmaps for total attached cells
def plot_heatmaps_total_attached(result_folder):
    total_attached_df = pd.read_csv(os.path.join(result_folder, 'total_attached_cells.csv'))

    # Ensure Flow_Speed and Adhesion_Strength values are of correct type
    total_attached_df['Flow_Speed'] = total_attached_df['Flow_Speed'].round().astype(int)
    total_attached_df['Adhesion_Strength'] = total_attached_df['Adhesion_Strength'].astype(float)

    # Get unique Flow_Speed and Adhesion_Strength values from the file
    flow_speeds = sorted(total_attached_df['Flow_Speed'].unique())
    adhesion_strengths = sorted(total_attached_df['Adhesion_Strength'].unique())

    print("Unique Flow_Speed values in DataFrame:", flow_speeds)
    print("Unique Adhesion_Strength values in DataFrame:", adhesion_strengths)

    # Group by parameters excluding Run_ID and average Total_Attached_Cells
    grouped_df = total_attached_df.groupby(
        ['Flow_Speed', 'Adhesion_Strength']
    )['Total_Attached_Cells'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 8))
    heatmap_data = grouped_df.pivot(index='Adhesion_Strength', columns='Flow_Speed', values='Total_Attached_Cells')

    cax = ax.imshow(heatmap_data, cmap='jet', aspect='auto', vmin=0, vmax=heatmap_data.max().max())
    ax.set_title("Total Attached Cells")
    ax.set_xlabel('Flow Speed')
    ax.set_ylabel('Adhesion Strength')
    ax.set_xticks(np.arange(len(flow_speeds)))
    ax.set_xticklabels(flow_speeds)
    ax.set_yticks(np.arange(len(adhesion_strengths)))
    ax.set_yticklabels(adhesion_strengths)

    fig.colorbar(cax, ax=ax)

    plt.savefig(os.path.join(result_folder, 'heatmap_total_attached_cells.pdf'))
    plt.show()
    plt.close(fig)

    print(f"Heatmap plotting completed. Results saved to {result_folder}")

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
    if isinstance(positions, str):
        positions_df = pd.read_csv(positions, compression='gzip')
    else:
        positions_df = pd.DataFrame(positions, columns=['Track_ID', 'Frame_Num', 'X_Position', 'Y_Position', 'Status'])

    max_frame = int(positions_df['Frame_Num'].max()) + 1

    if not fps:
        fps = max_frame / total_simulation_time

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (field_size[0], field_size[1]))

    # Define edge buffer based on cell diameter
    edge_buffer = int(cell_diameter_avg / 2)

    for frame_num in tqdm(range(max_frame), desc="Creating Video"):
        frame = np.ones((field_size[1], field_size[0], 3), dtype=np.uint8) * 255
        frame_positions = positions_df[positions_df['Frame_Num'] == frame_num]

        # Filter out positions that are too close to the edges
        active_positions = frame_positions[
            (frame_positions['X_Position'] >= edge_buffer) &
            (frame_positions['X_Position'] <= field_size[0] - edge_buffer) &
            (frame_positions['Y_Position'] >= edge_buffer) &
            (frame_positions['Y_Position'] <= field_size[1] - edge_buffer) &
            ((frame_positions['Status'] == 1) | (frame_positions['Status'] == 0))
        ]

        for _, row in active_positions.iterrows():
            x, y, status = int(row['X_Position']), int(row['Y_Position']), row['Status']
            color = (0, 0, 255) if status == 1 else (255, 0, 0)
            cv2.circle(frame, (x, y), edge_buffer, color, -1, lineType=cv2.LINE_AA)

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

# Function to format numeric values based on their type
def format_numeric_value(value):
    if isinstance(value, int):
        return str(value)
    elif isinstance(value, float):
        if value.is_integer():
            return str(int(value))  # Convert float to int if it has no decimal part
        else:
            return f"{value:.2f}".rstrip('0').rstrip('.')  # Format float with 2 decimal places, remove trailing zeros and dot
    else:
        return str(value)


def sum_attached_cells_over_time(result_folder, count_folder):
    # Load simulation parameters
    params_df = pd.read_csv(os.path.join(result_folder, 'simulation_parameters.csv'))

    # Generate a list of unique filenames based on parameters
    params_df['Filename'] = params_df.apply(
        lambda row: generate_filename(row['Flow_Speed'], row['Adhesion_Strength'], row['Cell_Density'], row['Run_ID'], row['Mask_Name']).replace('.csv', '_counts.csv'),
        axis=1
    )

    # Get only unique filenames
    unique_filenames = params_df['Filename'].unique()

    # Initialize the list to store the total attached cells
    total_attached_cells = []

    # Use tqdm to show progress while processing unique filenames
    for filename in tqdm(unique_filenames, desc="Collecting attached cells from the last row"):
        file_path = os.path.join(count_folder, filename)

        if os.path.exists(file_path):
            # Read the file containing arrested cells over time
            arrested_cells_df = pd.read_csv(file_path)

            # Get the value from the last row of the 'Attached_Count' column
            total_arrested_cells = arrested_cells_df['Attached_Count'].iloc[-1]

            # Extract the parameters associated with this filename (taking the first occurrence)
            row = params_df[params_df['Filename'] == filename].iloc[0]

            # Append the total number of arrested cells and simulation parameters to the list
            total_attached_cells.append({
                'Flow_Speed': row['Flow_Speed'],
                'Adhesion_Strength': row['Adhesion_Strength'],
                'Mask_Name': row['Mask_Name'],
                'Cell_Density': row['Cell_Density'],
                'Run_ID': row['Run_ID'],
                'Total_Attached_Cells': total_arrested_cells
            })
        else:
            print(f"File not found: {filename}")

    # Convert the list of total attached cells to a DataFrame
    total_attached_cells_df = pd.DataFrame(total_attached_cells)

    # Save the result to total_attached_cells.csv in the result folder
    output_path = os.path.join(result_folder, 'total_attached_cells.csv')
    total_attached_cells_df.to_csv(output_path, index=False)
    print(f"Total attached cells data saved to {output_path}")


def plot_arrested_cells(result_folder, cell_diameter_avg=10, field_size=(512, 512)):
    # Path where the position files are located (e.g., attached_cells folder)
    attached_cells_folder = os.path.join(result_folder, 'attached_cells')
    if not os.path.exists(attached_cells_folder):
        print(f"Attached cells folder not found: {attached_cells_folder}")
        return

    # Create a new folder to save the plots
    arrested_cells_plot_folder = os.path.join(result_folder, 'arrested_cells_plots')
    os.makedirs(arrested_cells_plot_folder, exist_ok=True)

    # Iterate through all files in the attached_cells folder
    for filename in os.listdir(attached_cells_folder):
        if filename.endswith('.csv'):
            # Load the position data
            file_path = os.path.join(attached_cells_folder, filename)
            positions_df = pd.read_csv(file_path)

            # Only plot if there are 2 or more attached cells
            if len(positions_df) < 2:
                print(f"Less than 2 attached cells found in {filename}. Skipping plot.")
                continue

            # Plot attached cells
            plt.figure(figsize=(6, 6))
            plt.xlim(0, field_size[0])
            plt.ylim(0, field_size[1])
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title(f"Attached Cells - {filename}")

            # Plot the positions of attached cells
            for _, row in positions_df.iterrows():
                plt.gca().add_patch(plt.Circle(
                    (row['X_Position'], row['Y_Position']),
                    cell_diameter_avg / 2,
                    color='red',
                    alpha=0.6
                ))

            plt.xlabel('X Position')
            plt.ylabel('Y Position')

            # Save plot as PNG
            plot_path = os.path.join(arrested_cells_plot_folder, f"{filename.replace('.csv', '')}_attached_cells.png")
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

            print(f"Attached cells plot saved to {plot_path}")
