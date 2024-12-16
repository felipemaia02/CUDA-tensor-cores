import pandas as pd
import matplotlib.pyplot as plt
import os

def load_and_process_csv(csv_file, script_filter):
    """
    Load and process the CSV file to generate a clean table for the specified script.
    
    Args:
        csv_file (str): Path to the CSV file.
        script_filter (str): Filter for the script type ('no_tensor_core' or 'with_tensor_core').
    
    Returns:
        pd.DataFrame: Processed table for the given script.
    """
    df = pd.read_csv(csv_file)
    
    # Filter only rows matching the script
    df_filtered = df[df['script'] == script_filter][['param_file', 'obs_file', 'initial_sum', 'final_sum', 'difference']]

    # Rename columns for clarity and rearrange the order
    df_filtered = df_filtered.rename(columns={
        'param_file': 'Parameter File',
        'obs_file': 'Obstacle File',
        'initial_sum': 'Initial Sum',
        'final_sum': 'Final Sum',
        'difference': 'Difference'
    })

    return df_filtered

def save_table_as_image(df, output_file):
    """
    Save the DataFrame as an image using Matplotlib.

    Args:
        df (pd.DataFrame): Table data to save.
        output_file (str): Path to the output image file.
    """
    _ , ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')  # Remove axes

    # Create the table
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#f0f0f0'] * df.columns.size
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Save the table as an image
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Table saved at: {output_file}")

def generate_execution_time_comparison(csv_file, output_file):
    """
    Generate a table comparing execution times between no_tensor_core and with_tensor_core.

    Args:
        csv_file (str): Path to the input CSV file.
        output_file (str): Path to save the table image.
    """
    df = pd.read_csv(csv_file)

    # Pivot the data to compare execution times
    pivot_df = df.pivot_table(
        values='execution_time',
        index=['param_file', 'obs_file'],
        columns='script'
    ).reset_index()

    # Rename columns for clarity
    pivot_df.columns = ['Parameter File', 'Obstacle File', 'No Tensor Core (s)', 'With Tensor Core (s)']

    # Save as image
    save_table_as_image(pivot_df, output_file)

def generate_sum_comparison_tables(csv_file, output_dir):
    """
    Generate two tables (images): one for no_tensor_core and another for with_tensor_core,
    as well as a table comparing execution times.

    Args:
        csv_file (str): Path to the input CSV file.
        output_dir (str): Directory to save the resulting images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate tables for both scripts
    script_configs = [
        {"filter": "no_tensor_core", "output": os.path.join(output_dir, "sum_comparison_no_tensor_core.png")},
        {"filter": "with_tensor_core", "output": os.path.join(output_dir, "sum_comparison_with_tensor_core.png")}
    ]

    for config in script_configs:
        df = load_and_process_csv(csv_file, config["filter"])
        if not df.empty:
            save_table_as_image(df, config["output"])
        else:
            print(f"No data available for {config['filter']}.")

    # Generate execution time comparison
    execution_time_output = os.path.join(output_dir, "execution_time_comparison.png")
    generate_execution_time_comparison(csv_file, execution_time_output)

    print(f"Execution time comparison table saved at: {execution_time_output}")
