# A Multi-Factor Adaptive Recalculation Strategy for Dynamic Shortest Path Maintenance

**Project:** M-FARS — Multi-Factor Adaptive Recalculation Strategy  
**Domain:** Algorithms and Data Structures (DAA), Dynamic Graph Algorithms  
**Language:** Python

## Overview

This project implements and evaluates a multi-factor adaptive strategy for maintaining shortest-path information in dynamic graphs.

In a dynamic graph, edge weights or graph conditions may change over time. Recomputing shortest paths after every update can be computationally expensive, particularly when updates occur frequently.

M-FARS explores an adaptive approach to deciding how shortest-path information should be recalculated as graph updates occur. The project includes algorithm versions, benchmarking scripts, result files, and visualization plots for performance analysis.

## Problem Statement

Traditional shortest-path algorithms, such as Dijkstra's algorithm, compute shortest paths for a given graph. When the graph changes, previously computed results may need to be updated.

The challenge is to reduce unnecessary recalculation work while maintaining correct shortest-path results.

This project investigates an adaptive recalculation strategy that considers multiple factors when determining how to handle graph updates.

## Objectives

- Implement the proposed multi-factor adaptive recalculation strategy.
- Compare its performance with a baseline shortest-path approach.
- Measure update-processing time and preprocessing overhead.
- Analyze work reduction and amortized performance.
- Study break-even update counts and strategy selection behavior.
- Visualize experimental results for easier interpretation.

## Methodology

The project contains multiple algorithm versions and benchmark scripts to support iterative development and evaluation.

The general experimental workflow is:

1. Initialize the graph and compute the required shortest-path information.
2. Apply graph updates according to the experiment setup.
3. Evaluate the adaptive recalculation strategy.
4. Compare the work and execution time with the baseline.
5. Record measurements in result files.
6. Generate plots to analyze performance across different graph sizes and update counts.

**Note:** The exact decision rules and mathematical formulation should be documented from the final algorithm implementation.

## Repository Structure

```text
Adaptive-Shortest-Path-Maintenance/
├── benchmark_v1.py
├── benchmark_v1_2.py
├── final_output.txt
├── m_fars_v1.py
├── m_fars_v1_1.py
├── m_fars_v1_2.py
├── m_fars_v1_3.py
├── m_fars_v1_4.py
├── m_fars_v1_5.py
├── m_fars_v1_5_final.py
├── m_fars_v1_5_plots.py
├── m_fars_v1_5_results.csv
├── m_fars_v1_5_break_even.csv
├── m_fars_v1_5_amortized_speedup.png
├── version1.py
└── plots/
    ├── 01_amortized_speedup_vs_updates.png
    ├── 02_mfars_vs_dijkstra_update_time.png
    ├── 03_preprocessing_cost_vs_graph_size.png
    ├── 04_work_reduction_vs_graph_size.png
    ├── 05_break_even_update_count_vs_graph_size.png
    └── 06_strategy_distribution_vs_updates.png
```

## Performance Analysis

The repository includes visualizations for the following experimental metrics:

| Plot | Purpose |
|---|---|
| Amortized speedup vs. updates | Examines performance as the number of updates changes |
| M-FARS vs. Dijkstra update time | Compares measured update-processing time |
| Preprocessing cost vs. graph size | Examines initialization overhead |
| Work reduction vs. graph size | Analyzes the change in computational work |
| Break-even update count vs. graph size | Examines when adaptive processing may offset preprocessing costs |
| Strategy distribution vs. updates | Shows how often different strategies are selected |

These plots are intended to support experimental analysis. Actual performance conclusions should be drawn from verified measurements and the benchmark configuration.

## Requirements

- Python 3
- Any third-party packages imported by the selected implementation or plotting scripts

Check the imports in the relevant Python files before installing dependencies. Add the confirmed dependencies to `requirements.txt` so the experiments can be reproduced.

## How to Run

Clone the repository:

```bash
git clone https://github.com/tandledisha/Adaptive_Shortest_Path_Maintenance.git
cd Adaptive_Shortest_Path_Maintenance
```

Run the final algorithm implementation:

```bash
python m_fars_v1_5_final.py
```

Run the benchmark scripts when their required inputs and configuration are available:

```bash
python benchmark_v1.py
python benchmark_v1_2.py
```

Generate plots using the plotting script if the required result files are available:

```bash
python m_fars_v1_5_plots.py
```

**Important:** Confirm the entry points, arguments, dependencies, and execution order in each script before running it. These commands are suggested starting points based on the filenames.

## Reproducibility

For reproducible experiments, document:

- Python version and dependency versions.
- Graph sizes and graph-generation method.
- Edge-update types and update counts.
- Baseline configuration.
- Number of repetitions and timing methodology.
- Exact commands used to generate the CSV files and plots.

## Author

**Disha**
