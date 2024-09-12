


# Function to calculate flow rate per frame
def calculate_flow_rate_per_frame(flow_speed, frame_interval):
    return flow_speed * frame_interval / PIXEL_SIZE  # Convert to pixels per frame


# Function to interpolate flow vector at a given cell position
def get_flow_vector(space, x, y):
    # Convert physical coordinates to grid indices
    i = int(x / space.dx)
    j = int(y / space.dy)

    # Ensure indices are within valid bounds
    i = min(max(i, 0), space.u_c.shape[1] - 1)
    j = min(max(j, 0), space.v_c.shape[0] - 1)

    # Directly read the velocity components from the grid
    u_direct = space.u_c[j, i]
    v_direct = space.v_c[j, i]

    return u_direct, v_direct

# Function to compute the average flow vector over the cell surface
def get_averaged_flow_vector(space, cell_position, cell_radius, num_samples=8):
    surface_points = sample_cell_surface(cell_position, cell_radius, num_samples)

    u_vectors = []
    v_vectors = []

    for point in surface_points:
        u_interp, v_interp = get_flow_vector(space, point[0], point[1])
        u_vectors.append(u_interp)
        v_vectors.append(v_interp)

    # Compute the average flow vector
    avg_u = np.mean(u_vectors)
    avg_v = np.mean(v_vectors)

    return avg_u, avg_v
