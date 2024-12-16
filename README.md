# CUDA-tensor-cores

This repository is designed for running and analyzing performance tests on CUDA scripts, comparing implementations **with Tensor Core** and **without Tensor Core**.

The project automates CUDA script executions, stores results in CSV files, and generates visualizations for data analysis.

---

## Project Structure

The repository is organized as follows:

```plaintext
CUDA-tensor-cores/
│
├── obstacles/                    # Obstacle files for execution
├── parameters/                   # Parameter files
├── results/                      # Generated results (CSV and visualizations)
│   ├── results.csv               # Execution results
│
├── src/                          # Main source code
│   ├── cuda/                     # CUDA implementation
│   │   ├── lb_cuda.cu            # CUDA code without Tensor Core
│   │   └── lb_cuda_tensor.cu     # CUDA code with Tensor Core
│   │
│   ├── cuda_test_functions.py    # Functions to execute tests
│   ├── generate_comparisons.py   # Generate comparison tables
│   ├── generate_visualizations.py # Generate visualizations (graphs and heatmaps)
│   ├── __init__.py               # Package initializer
│   └── main.py                   # Main script to execute the pipeline
│
├── .gitignore                    # Files ignored by Git
├── lb_cuda                       # CUDA binary (without Tensor Core)
├── lb_cuda_tensor                # CUDA binary (with Tensor Core)
└── README.md                     # This file
```

---

## Features

1. **Automated Execution of CUDA Scripts**:
   - **lb_cuda**: Implementation without Tensor Core.
   - **lb_cuda_tensor**: Implementation with Tensor Core.
   - Results are stored in the `results/results.csv` file.

2. **Comparison Tables**:
   - Comparison of **Initial Sum**, **Final Sum**, and **Difference** for each parameter and obstacle file.
   - Tables are generated as PNG images in the `results/visualizations` directory.

3. **Data Visualizations**:
   - **Heatmaps**: Displays differences between the results.
   - **Scatter Plots**: Shows the distribution of differences across obstacle files.

---

## Requirements

- **CUDA** (NVIDIA Toolkit)
- **Python 3.x**
- Required libraries:
  - `pandas`
  - `matplotlib`
  - `seaborn`

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/CUDA-tensor-cores.git
   cd CUDA-tensor-cores
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # For Linux/Mac
   .\.venv\Scripts\activate       # For Windows

   pip install -r requirements.txt
   ```

3. Ensure the CUDA binaries (`lb_cuda` and `lb_cuda_tensor`) are compiled and accessible in the root directory.

---

## Usage

### Step 1: Run the Main Script

The main script automates the entire pipeline:

- Executes CUDA scripts.
- Saves results to a CSV file.
- Generates comparison tables and visualizations.

Run:

```bash
python src/main.py
```

### Step 2: Visualize the Results

After running the main script, the following visualizations will be available in `results/visualizations`:

- **Comparison Tables**:
  - `sum_comparison_no_tensor_core.png`
  - `sum_comparison_with_tensor_core.png`

- **Heatmaps**:
  - `heatmap_no_tensor_core.png`
  - `heatmap_with_tensor_core.png`

- **Scatter Plots**:
  - `scatter_no_tensor_core.png`
  - `scatter_with_tensor_core.png`

---

## Example Workflow

1. **Run Tests**:
   - The tests execute CUDA scripts for each combination of parameters and obstacle files.

2. **Analyze Results**:
   - Comparison tables provide a clear view of numerical differences.
   - Heatmaps highlight variations in the results.
   - Scatter plots visually show the distribution of differences.

3. **Optimize CUDA Code**:
   - Use the results to evaluate the impact of Tensor Core optimizations.

---

## Contributing

Feel free to open an issue or submit a pull request if you'd like to contribute to this project.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

- **Author**: Felipe Maia
- **Email**: felipeoliveiramaia3@gmail.com
- **LinkedIn**: [Your LinkedIn](https://www.linkedin.com/in/felipeoliveira-maia/)
