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
def update_flow_field_with_phi_jax(cells, domain_size, grid_res, flow_speed):
    # Convert the attached cells into phi.jax obstacles (Sphere)
    obstacles = [Sphere(x=pos[0] * (grid_res[0] / domain_size[0]),  # Convert to grid coordinates
                        y=pos[1] * (grid_res[1] / domain_size[1]),  # Convert to grid coordinates
                        radius=cell_radius)
                 for pos in cells]

    # Define the boundary conditions for the phi.jax domain
    boundary = {
        'x-': vec(x=flow_speed, y=0),  # Apply the flow speed in the x-direction
        'x+': ZERO_GRADIENT,  # Outflow on the right (zero gradient for smooth exit)
        'y-': ZERO_GRADIENT,  # Open boundary at the bottom
        'y+': ZERO_GRADIENT   # Open boundary at the top
    }

    # Initialize the velocity grid in 2D with uniform flow
    velocity = StaggeredGrid((flow_speed, 0), boundary, x=grid_res[1], y=grid_res[0], bounds=Box(x=domain_size[1], y=domain_size[0]))
    pressure = None

    # Define the simulation step for phi.jax
    @jit_compile
    def step(v, p, dt=1.):
        v = advect.semi_lagrangian(v, v, dt)
        v = diffuse.explicit(v, viscosity, dt)

        v, p = fluid.make_incompressible(v, obstacles=obstacles, solve=Solve('CG', 1e-5, 1e-5))
        return v, p

    # Run the simulation for a few steps to reach a steady state
    for _ in tqdm(range(200), desc="Updating Flow Field"):
        velocity, pressure = step(velocity, pressure, dt=0.05)

    # Extract the updated flow components and convert to NumPy arrays
    u_np = velocity.staggered_tensor()[0].numpy(order='y,x')
    v_np = velocity.staggered_tensor()[1].numpy(order='y,x')

    return u_np, v_np
