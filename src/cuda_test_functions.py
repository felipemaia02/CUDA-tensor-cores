import os
import re
import subprocess
import csv

def extract_sums(output):
    """
    Extracts 'Initial sum' and 'End sum' values from the script output.
    """
    match_initial = re.search(r"Initial sum:\s*([\d\.\-e]+)", output)
    match_final = re.search(r"End sum:\s*([\d\.\-e]+)", output)
    if not match_initial or not match_final:
        raise ValueError("Could not find 'Initial sum' or 'End sum' in the output.")

    return float(match_initial.group(1)), float(match_final.group(1))

def extract_execution_time(output):
    """
    Extracts the execution time from the CUDA script output.
    Supports both standard CUDA and Tensor Core outputs.
    """
    match = re.search(r"(lb_2d_cuda(?:_tensor)?),([\d\.]+),", output)
    if not match:
        raise ValueError("Could not find the execution time in the output.")
    return float(match.group(2))

def run_test(script, param_file, obs_file, parameter_dir, obstacle_dir):
    """
    Runs the specified script with the given parameter and obstacle files.
    Returns execution time, initial sum, and final sum.
    """
    command = [script, os.path.join(parameter_dir, param_file), os.path.join(obstacle_dir, obs_file)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        output = result.stdout
        execution_time = extract_execution_time(output)
        initial_sum, final_sum = extract_sums(output)
        return execution_time, initial_sum, final_sum
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error running {command}: {e}")
    return None, None, None

def execute_tests(cuda_scripts, parameter_files, obstacle_files, repetitions, parameter_dir, obstacle_dir):
    """
    Runs all tests for the given scripts, parameter files, and obstacle files.
    Returns a list of results.
    """
    results = []
    for script_name, script_path in cuda_scripts.items():
        for param_file in parameter_files:
            for obs_file in obstacle_files:
                for rep in range(repetitions):
                    print(f"Running {script_name} with {param_file} and {obs_file} (repetition {rep + 1})...")
                    execution_time, initial_sum, final_sum = run_test(script_path, param_file, obs_file, parameter_dir, obstacle_dir)
                    if execution_time is None:
                        print(f"Skipping {param_file} and {obs_file} due to an error.")
                        continue
                    results.append({
                        "script": script_name,
                        "param_file": param_file,
                        "obs_file": obs_file,
                        "execution_time": execution_time,
                        "initial_sum": initial_sum,
                        "final_sum": final_sum,
                        "difference": initial_sum - final_sum,
                        "repetition": rep + 1
                    })
    return results

def save_results_to_csv(results, output_file):
    """
    Saves the test results to a CSV file.
    """
    with open(output_file, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["script", "param_file", "obs_file", "execution_time", "initial_sum", "final_sum", "difference", "repetition"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to {output_file}")
