# Investigating-the-Magnus-Effect---Year-1-Project
A Python simulation modeling the effect of the Magnus effect on the flight of a spinning ping pong ball. Includes a second script that compares the simulation results with experimental data collected from a DIY rotational launcher setup, visualizing trajectories and uncertainty bounds.

# Ping Pong Ball Flight Simulation with Magnus Effect

## Overview
This repository contains a Python simulation that models the flight of a spinning ping pong ball under the influence of the Magnus effect. The project also includes a comparison between the simulated trajectories and real-life experimental data collected using our DIY rotational launcher.  

## Features
- Simulates 3D flight of a ping pong ball with spin, accounting for drag and Magnus forces.  
- Includes a no-spin baseline for comparison.  
- Visualizes trajectories in 3D, top-down (X-Z), and side (X-Y) views.  
- Compares simulation with experimental data, including uncertainty estimates.  
- Generates plots and animations for trajectory analysis.  

## Usage
1. Install required Python packages:  
```bash
pip install numpy matplotlib scipy

```
2. To test the model using your own inputs:
```bash
python Magnus_force_model.py

```
3. To view our models prediction of our real lifes data and to see the comparison with the actual flight path of the ball:
```bash
python comparison_to_real_life_data.py

```
