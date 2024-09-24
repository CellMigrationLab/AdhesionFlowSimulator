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

from .io_utils import *  # Imports all functions and variables from io_utils
from .cell_dynamics import *  # Imports all functions and variables from cell_dynamics
from .flow_utils import *  # Imports all functions and variables from flow_utils

class Space:
    def __init__(self, field_size, grid_size, initial_flow_speed):
        self.field_size = field_size  # The physical size of the simulation field (in pixels)
        self.grid_size = grid_size  # The number of grid points in each dimension
        self.dx = field_size[0] / grid_size[0]  # Spacing in the x-direction (pixels per grid cell)
        self.dy = field_size[1] / grid_size[1]  # Spacing in the y-direction (pixels per grid cell)
        self.initial_flow_speed = initial_flow_speed

        # Initialize velocity grids (u_c for x-component, v_c for y-component)
        self.u_c = np.ones(grid_size) * initial_flow_speed  # x-component of velocity
        self.v_c = np.zeros(grid_size)  # y-component of velocity

    def reset_flow_field(self):
        """Reset the flow field to its initial conditions."""
        self.u_c = np.ones(self.grid_size) * self.initial_flow_speed
        self.v_c = np.zeros(self.grid_size)

def run_simulation(result_folder, field_size, grid_size, PIXEL_SIZE, time_steps, cell_diameter_avg, cell_radius, cell_area_avg, space, flow_rate_per_frame, flow_speed, adhesion_strength, cell_density, positions_folder, counts_folder, attachment_matrix_folder, flow_pattern_folder, attached_cells_folder, run_id, background, create_video=False, debug_mode=False, disable_flow_recompute=False, mask_name='uniform'):
    start_time = time.time()

    # Calculate the number of cells to introduce per step
    cells_per_step = calculate_cells_per_step(cell_density, field_size, flow_rate_per_frame, PIXEL_SIZE)
    total_cells = int(cells_per_step * time_steps) + 1

    print(f"Total cells to be introduced: {total_cells}")

    # Initialize cells array with an additional column for Track_ID
    cells = np.zeros((total_cells, 7))  # Each cell: [Track_ID, x_position, y_position, attached, active, u_direction, v_direction]
    cells[:, 0] = np.arange(total_cells)  # Assign unique Track_ID to each cell
    cells[:, 1] = -1  # Initialize x_position to -1 (out of bounds)
    cells[:, 2] = -1  # Initialize y_position to -1 (out of bounds)

    attached_positions = []
    attached_counts = []
    all_positions = []  # List to store all cell positions and statuses for each frame

    current_cell_idx = 0  # This index will track which cells have been introduced

    # Accumulator for probability when cells_per_step < 1
    accumulated_prob = 0

    # Set up the figure for debugging mode
    fig = None
    if debug_mode:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, field_size[0])
        ax.set_ylim(0, field_size[1])

    for frame_num in tqdm(range(time_steps), desc=f"Simulation for flow_speed={flow_speed}, adhesion_strength={adhesion_strength}, mask_name={mask_name}, cell_density={cell_density}, run_id={run_id}"):
        # Calculate the number of cells to add in this frame
        if cells_per_step < 1:
            accumulated_prob += cells_per_step
            if accumulated_prob >= 1:
                num_new_cells = 1
                accumulated_prob -= 1  # Reduce accumulated probability by 1 after adding a cell
            else:
                num_new_cells = 0
        else:
            num_new_cells = int(cells_per_step)

        # Introduce new cells if there are still cells left to introduce
        if current_cell_idx < total_cells and num_new_cells > 0:
            added_cells = introduce_new_cells(cells, current_cell_idx, num_new_cells, cells, cell_radius, field_size)
            current_cell_idx += added_cells

        new_attachment_occurred = False  # Flag to track if any new cell was attached

        for i in range(len(cells)):
            if cells[i, 3] == 1:  # If cell is already attached, do nothing
                continue

            if cells[i, 4] == 0:  # If cell is already deactivated, do nothing
                continue

            # Calculate the attachment probability
            u_interp, v_interp = get_averaged_flow_vector(space, cells[i, 1:3], cell_radius)  # Get the flow vector for movement

            attachment_prob = np.random.uniform(0, 1)
            attachment_rate = calculate_attachment_probability(np.linalg.norm([u_interp, v_interp]), adhesion_strength, background, cells[i, 1:3], cell_diameter_avg, field_size)

            if attachment_prob < attachment_rate:
                # Attach the cell
                cells[i, 3] = 1
                attached_positions.append(cells[i, 1:3].copy())  # Record the position
                #print(f"Cell {i} attached at position x ({cells[i, 1]}, position y  {cells[i, 2]})")
                new_attachment_occurred = True  # Set flag to True if a cell attached
            else:
                # Move the cell according to the flow vector
                cells[i, 1] += u_interp
                cells[i, 2] += v_interp

                # Check boundaries and deactivate the cell if it leaves the field
                if cells[i, 1] >= field_size[0] or cells[i, 2] >= field_size[1] or cells[i, 1] < 0 or cells[i, 2] < 0:
                    cells[i, 4] = 0  # Deactivate cell if it leaves the field

        if new_attachment_occurred and not disable_flow_recompute:
            print(f"Updating flow field after frame {frame_num}...")
            u_center, v_center = update_flow_field_with_phi_jax(attached_positions, cell_radius, field_size, grid_size, flow_rate_per_frame)
            space.u_c = u_center
            space.v_c = v_center

            # Plot the updated space vector field after the cell attaches
            plt.figure(figsize=(6, 6))
            plt.streamplot(np.arange(space.u_c.shape[0]), np.arange(space.u_c.shape[1]), space.u_c, space.v_c, density=1.0)

            # Plot all attached cells
            attached_cell_positions = np.array(attached_positions)
            plt.scatter(attached_cell_positions[:, 0], attached_cell_positions[:, 1], color='red', s=cell_area_avg/2, label='Attached Cells')

            plt.title(f"Flow Field After Frame {frame_num}")
            plt.gca()
            plt.legend()

            # Save the plot
            output_filename = generate_filename(flow_speed, adhesion_strength, mask_name, cell_density, run_id).replace('.csv', f'_flow_field_frame_{frame_num}.png')
            plt.savefig(os.path.join(flow_pattern_folder, output_filename))

            plt.show()  # Show the plot after saving
            plt.close()  # Close the plot to free memory

        # Debug mode: Display each frame
        if debug_mode:
            ax.clear()
            ax.set_xlim(0, field_size[0])
            ax.set_ylim(0, field_size[1])
            colors = ['red' if status == 1 else 'blue' for status in cells[:, 3]]
            ax.scatter(cells[:, 1], cells[:, 2], c=colors, s=cell_area_avg/2, edgecolors='k')
            plt.pause(0.01)  # Pause to update the display
            clear_output(wait=True)
            display(fig)

        # Save positions for this frame
        frame_positions = np.column_stack((cells[:, 0], np.full(len(cells), frame_num), cells[:, 1:4]))  # Include Track_ID and Frame_Num
        all_positions.append(frame_positions)
        attached_counts.append(np.sum(cells[:, 3] == 1))

    # Save positions to a compressed CSV file with gz compression
    all_positions_np = np.vstack(all_positions)
    positions_df = pd.DataFrame(all_positions_np, columns=['Track_ID', 'Frame_Num', 'X_Position', 'Y_Position', 'Status'])
    positions_df.to_csv(os.path.join(positions_folder, generate_filename(flow_speed, adhesion_strength, cell_density, run_id, mask_name).replace('.csv', '_positions.csv.gz')), index=False, compression='gzip')

    # Save attached positions to CSV file
    attached_positions_df = pd.DataFrame(attached_positions, columns=['X_Position', 'Y_Position'])
    attached_positions_df.to_csv(os.path.join(attached_cells_folder, generate_filename(flow_speed, adhesion_strength, cell_density, run_id, mask_name).replace('.csv', '_attached_positions.csv')), index=False)

    # Save the number of attached cells per frame
    attached_counts_df = pd.DataFrame(attached_counts, columns=['Attached_Count'])
    attached_counts_df.to_csv(os.path.join(counts_folder, generate_filename(flow_speed, adhesion_strength, cell_density, run_id, mask_name).replace('.csv', '_counts.csv')), index=False)

    # Save the attachment matrix as an image
    plt.imshow(background, cmap='viridis', origin='lower', aspect='auto')
    plt.colorbar(label='Attachment Probability')
    plt.title(f'Attachment Matrix for flow_speed={flow_speed}, adhesion_strength={adhesion_strength}, mask_name={mask_name}, cell_density={cell_density}, run_id={run_id}')
    plt.savefig(os.path.join(attachment_matrix_folder, generate_filename(flow_speed, adhesion_strength, cell_density, run_id, mask_name).replace('.csv', '_matrix.png')))
    plt.close()

    if create_video:
        create_video_from_positions(all_positions_np, os.path.join(positions_folder, generate_filename(flow_speed, adhesion_strength, cell_density, run_id, mask_name).replace('.csv', '.mp4')), field_size, cell_diameter_avg, fps=25)

    # Save parameters
    save_parameters(result_folder, flow_speed, adhesion_strength, cell_density, run_id, mask_name)

    elapsed_time = time.time() - start_time
    print(f'Simulation for flow_speed={flow_speed}, adhesion_strength={adhesion_strength}, mask_name={mask_name}, cell_density={cell_density}, run_id={run_id} took {elapsed_time:.2f} seconds.')

    # Clean up the space at the end of the simulation
    space.reset_flow_field()
    return len(attached_positions)  # Return the total number of attached positions
