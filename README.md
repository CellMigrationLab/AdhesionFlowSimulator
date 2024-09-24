# Adhesion Flow Simulator


The **Adhesion Flow Simulator** is a computational tool designed to simulate and visualize the behavior of cells under the influence of fluid flow. This pipeline models cell attachment to endothelial surfaces, guided by factors like flow speed, adhesion strength, and receptor map data. The simulator integrates biophysical properties of cells and fluid dynamics, offering insights into cell behavior in various conditions.

The simulations use real-time flow adjustments and cell attachment dynamics to generate data, which can be visualized through heatmaps, flow fields, and Ripley’s L function analysis.

| Example of Simulation | Example of Flow Field computation |
|----------------|-----------|
| <img src="https://github.com/CellMigrationLab/AdhesionFlowSimulator/blob/main/images/example.gif" width="300px" alt="Example simulation"> | <img src="https://github.com/CellMigrationLab/AdhesionFlowSimulator/blob/main/images/flow_field.png?raw=true" width="300px" alt="PNG image"> |

## Features

- **Flow-Driven Simulations**: Simulate cell movement and attachment in a fluid environment with tunable flow speeds.
- **Receptor Map Support**: Load receptor map images to introduce spatial heterogeneity in cell adhesion probabilities.
- **Ripley’s L Function**: Analyze spatial point patterns using Ripley’s L function to assess clustering of attached cells.
- **Heatmaps and Visualization**: Generate visual outputs like flow fields, heatmaps, and cell trajectory videos.
- **Checkpointing**: Save and resume simulations from checkpoints to handle large-scale runs efficiently.
