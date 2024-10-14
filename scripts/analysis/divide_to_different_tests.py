import numpy as np
import pandas as pd

# Load the Excel file
df = pd.read_excel("results/results_dudu_231024.xlsx")

df['Run Type (CPU/GPU)'] = df['Run Type']
# Split the "Test Name" column into "Run Type" and "Test Name"
df['Run Type'] = df['Test Name'].str.split('_').str[0]  # Extract the first part (Run Type)
df['Test Name'] = df['Test Name'].str.split('_').str[1:].apply('_'.join)  # Rejoin the remaining parts as Test Name

# Get the unique test names
unique_tests = df['Test Name'].unique()

# Create an Excel writer to save multiple sheets
with pd.ExcelWriter("results/results_dudu_231024_with_sheets.xlsx") as writer:
    for test_name in unique_tests:
        # Filter the DataFrame for each unique test name
        test_df = df[df['Test Name'] == test_name]
        
        # Write each filtered DataFrame to a separate sheet
        test_df.to_excel(writer, sheet_name=test_name, index=False)

# Output to show the process is complete
print(f"New Excel file with separate sheets for each test type has been created.")

