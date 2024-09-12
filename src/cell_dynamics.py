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



def calculate_attachment_probability(flow_rate, adhesion_strength, background, cell_position):
    """
    Simulate attachment probability considering flow and adhesion strength of cancer cells,
    with local endothelial adhesion strength represented by the background map.

    Args:
        flow_rate (float): The flow rate affecting the attachment probability.
        cancer_cell_adhesion_strength (float): Adhesion strength of cancer cells.
        background (np.ndarray): The receptor map or background field representing endothelial adhesion strength.
        cell_position (tuple): The (x, y) position of the cell in the field.

    Returns:
        float: The adjusted attachment probability for the cell.
    """
    # Calculate base attachment probability using cancer cell adhesion strength
    base_prob = adhesion_strength
    x, y = int(cell_position[0]), int(cell_position[1])
    radius = int(cell_diameter_avg / 2)  # Assuming integer radius

    # Boundary checks to prevent out-of-bounds indexing
    if x - radius < 0 or x + radius >= field_size[0] or y - radius < 0 or y + radius >= field_size[1]:
        return 0  # No attachment if the cell is out of bounds

    # Extract the local background area covered by the cell
    local_background = background[max(0, x-radius):min(field_size[0], x+radius+1), max(0, y-radius):min(field_size[1], y+radius+1)]

    # Calculate the average endothelial adhesion strength from the local background
    mean_endothelial_adhesion = np.mean(local_background)

    # Flow effect reduces the attachment probability exponentially with increasing flow
    flow_effect = np.exp(-flow_rate)

    # Final attachment probability adjusted by local endothelial adhesion
    adjusted_prob = base_prob * flow_effect * mean_endothelial_adhesion

    return adjusted_prob

def calculate_cells_per_step(cell_density, field_size, flow_rate_per_frame):
    # Calculate the area covered per frame in pixels²
    area_covered_per_frame = flow_rate_per_frame * field_size[1]  # field_size[1] is height in pixels

    # Convert cell density from cells per micron² to cells per pixel²
    cell_density_per_pixel = cell_density * (PIXEL_SIZE ** 2)

    # Calculate the number of cells to introduce per step (can be fractional)
    cells_per_step = cell_density_per_pixel * area_covered_per_frame

    return cells_per_step


# Function to introduce new cells ensuring no overlap and avoiding edges (cells must start at x = 0)
def introduce_new_cells(cells, current_cell_idx, num_cells_to_add, existing_cells, cell_radius, central_field_size):
    new_cells_idx = np.arange(current_cell_idx, min(current_cell_idx + num_cells_to_add, len(cells)))
    for idx in new_cells_idx:
        position = find_non_overlapping_position(existing_cells, cell_radius, central_field_size)
        if position is not None:
            cells[idx, 1:3] = position  # Set x and y positions
            cells[idx, 4] = 1  # Mark as active
            cells[idx, 5:7] = [1, 0]  # Initialize direction vector aligned with the flow (rightward)
    return len(new_cells_idx)

# Function to check if two cells overlap
def check_overlap(pos1, pos2, cell_radius):
    distance = np.linalg.norm(pos1 - pos2)
    return distance < 2 * cell_radius

# Function to find a non-overlapping position for a new cell
def find_non_overlapping_position(existing_cells, cell_radius, field_size):
    max_attempts = 100  # Limit the number of attempts to prevent infinite loops
    for _ in range(max_attempts):
        new_position = np.array([0, np.random.uniform(0, field_size[1])])  # Start at x = 0, random y
        if all(not check_overlap(new_position, cell[1:3], cell_radius) for cell in existing_cells if cell[4] == 1):
            return new_position
    return None  # Return None if a non-overlapping position couldn't be found


# Function to sample points around the perimeter of the cell
def sample_cell_surface(cell_position, cell_radius, num_samples=8):
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False)
    surface_points = np.array([
        [cell_position[0] + cell_radius * np.cos(angle),
         cell_position[1] + cell_radius * np.sin(angle)]
        for angle in angles
    ])
    return surface_points
