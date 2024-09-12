import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from tqdm.notebook import tqdm

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

# Function to compute Ripley's L function for each simulation
def compute_ripley_l(result_folder, radii):
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

        if attached_positions.shape[0] > 0:
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
        if attached_positions.shape[0] > 0:
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
