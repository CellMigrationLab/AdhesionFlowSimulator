import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from tqdm.notebook import tqdm

from .io_utils import *  # Imports all functions and variables from io_utils
from .flow_utils import *  # Imports all functions and variables from flow_utils


# Define Ripley's K function
def ripley_k(points, r, area):
    n = len(points)
    d_matrix = distance_matrix(points, points)
    sum_indicator = np.sum(d_matrix < r) - n  # Subtract n to exclude self-pairs
    K_r = (area / (n ** 2)) * sum_indicator
    return K_r

# Define Ripley's L function
def ripley_l(points, r, area):
    K_r = ripley_k(points, r, area)
    L_r = np.sqrt(K_r / np.pi) - r
    return L_r

def plot_heatmaps_ripley_l(result_folder, radius=51):
    # Load Ripley L values
    ripley_l_df = pd.read_csv(os.path.join(result_folder, 'Ripley_L_Values.csv'))

    # Ensure Flow_Speed and Adhesion_Strength values are of correct type
    ripley_l_df['Flow_Speed'] = ripley_l_df['Flow_Speed']
    ripley_l_df['Adhesion_Strength'] = ripley_l_df['Adhesion_Strength']

    # Filter the data for the specific radius
    filtered_df = ripley_l_df[ripley_l_df['Radius'] == radius]

    if filtered_df.empty:
        print(f"No data available for radius {radius}.")
        return

    # Remove rows where L_Value is NaN for counting, but keep the full DataFrame for plotting
    cleaned_df = filtered_df.dropna(subset=['L_Value'])

    # Group by parameters and calculate the average L_Value (after removing NaN)
    grouped_df = filtered_df.groupby(
        ['Flow_Speed', 'Adhesion_Strength']
    )['L_Value'].mean().reset_index()

    # Count the number of values for each combination of Flow_Speed and Adhesion_Strength, excluding NaN
    count_per_condition = cleaned_df.groupby(['Flow_Speed', 'Adhesion_Strength']).size().reset_index(name='Count')

    # Print the counts for each condition
    print("Number of values per condition (Flow Speed, Adhesion Strength), excluding NaN values:")
    print(count_per_condition)

    # Check if there are fewer than 10 values for any condition
    problematic_conditions = count_per_condition[count_per_condition['Count'] < 10]

    # If any conditions have fewer than 10 values, print a specific warning
    if not problematic_conditions.empty:
        print(f"Warning: The following conditions have fewer than 10 values for radius {radius}:")
        for _, row in problematic_conditions.iterrows():
            flow_speed, adhesion_strength, count = row['Flow_Speed'], row['Adhesion_Strength'], row['Count']
            print(f"  - Flow Speed: {flow_speed}, Adhesion Strength: {adhesion_strength}, Count: {count}")

    # Save the cleaned DataFrame (after removing NaNs) for further analysis
    cleaned_df.to_csv(os.path.join(result_folder, f'cleaned_ripley_l_values_radius_{radius}.csv'), index=False)
    print(f"Final DataFrame (after removing NaNs) saved to {result_folder}")

    # Pivot the grouped DataFrame (with averages) for heatmap
    heatmap_data = grouped_df.pivot(
        index='Adhesion_Strength', columns='Flow_Speed', values='L_Value'
    )

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    # Set background color of the axes
    ax.set_facecolor('yellow')

    # Plot the heatmap with chosen colormap, fixed range -50 to +50
    cax = ax.imshow(heatmap_data, cmap='coolwarm', aspect='auto', vmin=-50, vmax=50)

    ax.set_title(f"Ripley's L Function Heatmap at Radius {radius}")
    ax.set_xlabel('Flow Speed')
    ax.set_ylabel('Adhesion Strength')
    ax.set_xticks(np.arange(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticks(np.arange(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)

    # Add colorbar
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label("L Value")

    # Save heatmap
    plt.savefig(os.path.join(result_folder, f'heatmap_ripley_l_values_radius_{radius}.pdf'))
    plt.show()
    plt.close(fig)

    print(f"Heatmap plotting completed for radius {radius}. Results saved to {result_folder}")

# Function to compute Ripley's L function for each simulation
def compute_ripley_l(result_folder, radii, field_size, FRAME_INTERVAL, PIXEL_SIZE):
    attached_cells_folder = os.path.join(result_folder, 'attached_cells')
    ripley_folder = os.path.join(result_folder, 'ripley_folder')
    os.makedirs(ripley_folder, exist_ok=True)

    params_df = pd.read_csv(os.path.join(result_folder, 'simulation_parameters.csv'))

    # Prepare to collect data to avoid multiple pd.concat calls
    rows = []

    # Generate a list of expected filenames
    expected_filenames = [
        generate_filename(row['Flow_Speed'], row['Adhesion_Strength'], row['Cell_Density'], row['Run_ID'], row['Mask_Name']).replace('.csv', '_attached_positions.csv')
        for _, row in params_df.iterrows()
    ]

    expected_filenames = list(set(expected_filenames))

    for filename in tqdm(expected_filenames, desc="Computing Ripley's L"):
        file_path = os.path.join(attached_cells_folder, filename)
        print(f"Processing file: {filename}")

        # Load attached positions
        try:
            attached_positions = np.loadtxt(file_path, delimiter=",", skiprows=1)  # Skip header
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            attached_positions = np.empty((0, 2))

        area = field_size[0] * field_size[1]

        # Ensure attached_positions is 2D with 2 columns
        if attached_positions.ndim == 1 and attached_positions.size % 2 == 0:
            attached_positions = attached_positions.reshape(-1, 2)
        elif attached_positions.ndim != 2 or attached_positions.shape[1] != 2:
            attached_positions = np.empty((0, 2))

        # Check if at least two cells are attached before computing Ripley's L
        if attached_positions.shape[0] >= 2:
            L_values = [ripley_l(attached_positions, r, area) for r in radii]
        else:
            L_values = [np.nan] * len(radii)

        # Retrieve parameters from params_df
        filename_without_extension = filename.replace('_attached_positions.csv', '.csv')
        if filename_without_extension in params_df['Filename'].values:
            params = params_df[params_df['Filename'] == filename_without_extension].iloc[0]
            flow_speed = float(params['Flow_Speed'])
            flow_rate_per_frame = calculate_flow_rate_per_frame(flow_speed, FRAME_INTERVAL)
            adhesion_strength = float(params['Adhesion_Strength'])
            mask_name = params['Mask_Name']
            cell_density = float(params['Cell_Density'])
            run_id = float(params['Run_ID'])
        else:
            continue  # Skip this file if parameters are missing

        # Collect results in rows list
        for r_index, r in enumerate(radii):
            rows.append({
                'Flow_Speed': flow_speed,
                'Flow_Rate_Per_Frame': flow_rate_per_frame,
                'Adhesion_Strength': adhesion_strength,
                'Mask_Name': mask_name,
                'Cell_Density': cell_density,
                'Run_ID': run_id,
                'Radius': r,
                'L_Value': L_values[r_index],
                'Filename': filename
            })

        # Plot Ripley's L function
        if attached_positions.shape[0] >= 2:
            plt.figure(figsize=(10, 5))
            plt.plot(radii, L_values, label=f"L(r) for Flow {flow_speed} μm/s, Adhesion {adhesion_strength}, Mask_Name {mask_name}, Density {cell_density}, Run {run_id}")
            plt.xlabel('Radius (r) (micrometers)')
            plt.ylabel("Ripley's L Function")
            plt.title(f"Ripley's L Function - Flow {flow_speed} μm/s, Adhesion {adhesion_strength}, Mask_Name {mask_name}, Density {cell_density}, Run {run_id}")
            plt.legend()
            plt.grid(True)

            pdf_path = os.path.join(ripley_folder, f'ripley_L_flow_{format_numeric_value(flow_speed)}_as_{format_numeric_value(adhesion_strength)}_at_{mask_name}_cd_{format_numeric_value(cell_density)}_run_{format_numeric_value(run_id)}.pdf')
            plt.savefig(pdf_path, bbox_inches='tight')
            plt.close()  # Free memory by closing the figure

    # Create a DataFrame from the collected rows and save it
    l_values_df = pd.DataFrame(rows)
    l_values_df.to_csv(os.path.join(result_folder, 'Ripley_L_Values.csv'), index=False)
    print(f"Ripley's L function computation completed. Results saved to {result_folder}")
