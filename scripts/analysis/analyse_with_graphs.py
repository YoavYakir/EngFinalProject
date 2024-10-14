import pandas as pd
import matplotlib.pyplot as plt
import os

test_name = 'train'
# Create the directory for saving the plots if it doesn't exist
current_dir = os.getcwd()
save_path = os.path.join(current_dir, "results", "Figs", test_name)
os.makedirs(save_path, exist_ok=True)

# Load the Excel file
test_df = pd.read_excel("results/results_dudu_161024_with_sheets.xlsx",sheet_name =test_name)

# Define the target parameters and the test to analyze (iterative_test)
target_params = ['Elapsed Time', 'Average Memory Usage (%)', 'Average Memory Usage (MB)', 'Average Cache Usage (MB)']

# Plot for each Run Type
run_types = test_df['Run Type'].unique()

for param in target_params:
    plt.figure(figsize=(10, 6))
    for run_type in run_types:
        subset = test_df[test_df['Run Type'] == run_type]
        plt.plot(subset['Batch Size'], subset[param], label=run_type, marker='o')
    
    # Set plot details
    plt.title(f'Effect of Batch Size on {param} for {test_name}')
    plt.xlabel('Batch Size')
    plt.ylabel(param)
    plt.legend(title='Run Type')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f"results/Figs/{test_name}/{param}_batch_size_{test_name}.png")

# Group by 'Run Type' and calculate the mean for each target parameter
average_df = test_df.groupby('Run Type')[target_params].mean()

# Create a separate bar chart for each target parameter
for param in target_params:
    plt.figure(figsize=(10, 6))
    average_df[param].plot(kind='bar', color='skyblue')
    
    # Set plot details
    plt.title(f'Average {param} Comparison for {test_name}')
    plt.ylabel(f'Average {param}')
    plt.grid(True)
     # Set y-axis limits to zoom in, particularly for memory usage parameters
    if 'Memory' in param:
        plt.ylim([average_df[param].min() * 0.95, average_df[param].max() * 1.05])  # Zoom in for memory-related parameters
    else:
        plt.ylim([average_df[param].min() * 0.9, average_df[param].max() * 1.1])  # General zoom for other parameters
    plt.tight_layout()
    
    # Save the plot as an image file instead of showing it
    plt.savefig(f"results/Figs/{test_name}/average_comparison_{param}_{test_name}.png")