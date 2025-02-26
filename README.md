# Relative Performance Analysis

## Overview
This Jupyter Notebook analyzes the relative performance of different sectors based on Jensen's Alpha. It calculates the alpha for various lookback periods and compares performance over different time windows.

## Features
- Computes Jensen's Alpha using historical data for multiple sectors
- Supports configurable lookback periods and time windows
- Visualizes sector performance using Matplotlib
- Highlights trends and movements in sector performance over time

## Requirements
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas matplotlib
```

## How to Use
1. Load the required historical data into `history_df`.
2. Define the sectors and their respective identifiers in `sector_dict`.
3. Run the notebook to compute Jensen's Alpha.
4. The notebook generates a plot showing the relative performance of sectors over different time frames.

## Customization
- Modify `windows` to change the time periods used for analysis.
- Adjust `lookback_steps` to refine the granularity of the performance comparison.
- Customize `sector_colors` for improved visualization.

## Output
- A scatter plot illustrating the movement of sector performance.
- A legend mapping each sector to its respective color.

