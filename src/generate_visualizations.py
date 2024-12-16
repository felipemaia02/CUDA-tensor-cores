import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(csv_file, tensor_core):
    """
    Loads the CSV file and filters data for the specified tensor core condition.
    """
    df = pd.read_csv(csv_file)
    return df[df['script'] == tensor_core]

def generate_heatmap(df, title, output_file):
    """
    Generates a heatmap of the difference values.
    """
    pivot_df = df.pivot(index='param_file', columns='obs_file', values='difference')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="coolwarm", cbar_kws={'label': 'Difference'})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Heatmap saved at: {output_file}")

def generate_scatter_plot(df, title, output_file):
    """
    Generates a scatter plot of the differences.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='obs_file', y='difference', hue='param_file', style='param_file', s=100)
    plt.title(title)
    plt.xlabel("Obstacle File")
    plt.ylabel("Difference")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"Scatter Plot saved at: {output_file}")

def generate_all_visualizations(csv_file, output_dir):
    """
    Orchestrates the generation of heatmaps and scatter plots for both conditions.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data for both tensor core conditions
    df_no_tensor = load_data(csv_file, "no_tensor_core")
    df_with_tensor = load_data(csv_file, "with_tensor_core")

    # Generate Heatmaps
    generate_heatmap(df_no_tensor, "Difference Heatmap (no_tensor_core)", 
                     os.path.join(output_dir, "heatmap_no_tensor_core.png"))
    generate_heatmap(df_with_tensor, "Difference Heatmap (with_tensor_core)", 
                     os.path.join(output_dir, "heatmap_with_tensor_core.png"))

    # Generate Scatter Plots
    generate_scatter_plot(df_no_tensor, "Scatter Plot (no_tensor_core)", 
                          os.path.join(output_dir, "scatter_no_tensor_core.png"))
    generate_scatter_plot(df_with_tensor, "Scatter Plot (with_tensor_core)", 
                          os.path.join(output_dir, "scatter_with_tensor_core.png"))
