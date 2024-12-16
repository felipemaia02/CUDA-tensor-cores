import os
from src.cuda_test_functions import execute_tests, save_results_to_csv
from src.generate_comparisons import generate_sum_comparison_tables
from src.generate_visualizations import generate_all_visualizations

# Configuration
CUDA_SCRIPTS = {
    "no_tensor_core": "./lb_cuda",
    "with_tensor_core": "./lb_cuda_tensor"
}
PARAMETER_FILES = ["anb.par", "anb.par2"]
OBSTACLE_FILES = ["anb.obs8", "anb.obs11", "anb.obs12", "anb.obs31", "anb.obs32", "anb.obs.41", "anb.obs90"]
REPETITIONS = 1

# Directories
PARAMETER_DIR = "parameters"
OBSTACLE_DIR = "obstacles"
RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")

def setup_directories():
    """
    Ensures that the results directory exists.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

def run_tests():
    """
    Executes the CUDA tests and saves the results to a CSV file.
    """
    print("Executing tests...")
    results = execute_tests(
        CUDA_SCRIPTS,
        PARAMETER_FILES,
        OBSTACLE_FILES,
        REPETITIONS,
        PARAMETER_DIR,
        OBSTACLE_DIR
    )
    save_results_to_csv(results, RESULTS_CSV)
    print(f"Results saved to {RESULTS_CSV}")

def generate_comparison_table():
    """
    Generates the sum comparison table image.
    """
    print("Generating comparison table...")
    generate_sum_comparison_tables(RESULTS_CSV, RESULTS_DIR)
    print(f"Comparison table saved to {RESULTS_DIR}")

def generate_visualizations():
    """
    Generates visualizations (heatmaps and scatter plots) from the results CSV.
    """
    print("Generating visualizations...")
    generate_all_visualizations(RESULTS_CSV, RESULTS_DIR)
    print("Visualizations generated successfully.")

def main():
    """
    Main function orchestrating the workflow: 
    1. Setup directories
    2. Run tests
    3. Generate comparison table
    4. Generate visualizations
    """
    setup_directories()
    run_tests()
    generate_comparison_table()
    generate_visualizations()

if __name__ == "__main__":
    main()
