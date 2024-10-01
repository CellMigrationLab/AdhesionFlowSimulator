# Adhesion Flow Simulator

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13835127.svg)](https://doi.org/10.5281/zenodo.13835127)

The **Adhesion Flow Simulator** is a computational tool designed to simulate the behavior of cells under under flow. We use this pipeline models cell attachment to endothelial surfaces, guided by factors like flow speed, adhesion strength, and receptor map data. The simulator integrates biophysical properties of cells and fluid dynamics, offering insights into cell behavior in various conditions.

The simulations use real-time flow adjustments and cell attachment dynamics to generate data, which can be visualized through heatmaps, flow fields, and Ripley’s L function analysis.

| Example of Simulation | Example of Flow Field computation |
|----------------|-----------|
| <img src="https://github.com/CellMigrationLab/AdhesionFlowSimulator/blob/main/images/example.gif" width="300px" alt="Example simulation"> | <img src="https://github.com/CellMigrationLab/AdhesionFlowSimulator/blob/main/images/flow_field.png?raw=true" width="300px" alt="PNG image"> |

## Features

- **Flow-Driven Simulations**: Simulate cell movement and attachment in fluid environments with customizable flow speeds and fluid properties.
- **Receptor Map Support**: Load receptor map images, such as CD44 stainings, to introduce spatial heterogeneity in cell adhesion probabilities across the endothelial surface.
- **Ripley’s L Function**: Quantitatively assess spatial clustering of attached cells using Ripley’s L function, allowing for in-depth analysis of clustering behavior.
- **Heatmaps and Visualization**: Automatically generate visual outputs such as heatmaps of attached cells, flow fields, and simulation videos showcasing cell trajectories.
- **Static vs. Dynamic Flow Fields**: Choose between static flow conditions, where flow remains constant, or dynamic conditions, where flow is updated after each cell attachment to account for changes in fluid dynamics.
- **Checkpointing**: Save simulation progress and resume large-scale runs with minimal computational overhead, ensuring efficient handling of extensive simulations.

## Quick Start

You can quickly try out the **Adhesion Flow Simulator** using Google Colab:
[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CellMigrationLab/AdhesionFlowSimulator/blob/refs/tags/v0.1/notebooks/AdhesionFlowSimulator.ipynb)

## Acknowledgments

The [phiFlow Python](https://github.com/tum-pbs/PhiFlow) package is used for solving the Navier-Stokes equations in dynamic flow simulations.

## Citation

**Fast label-free live imaging reveals key roles of flow dynamics and CD44-HA interaction in cancer cell arrest on endothelial monolayers**  
Gautier Follain, Sujan Ghimire, Joanna W Pylvänäinen, Monika Vaitkevičiūtė, Diana H Wurzinger, Camilo Guzmán, James RW Conway, Michal Dibus, Sanna Oikari, Kirsi Rilla, Marko Salmi, Johanna Ivaska, Guillaume Jacquemet  
*bioRxiv* (2024). doi: [10.1101/2024.09.30.615654](https://doi.org/10.1101/2024.09.30.615654)




