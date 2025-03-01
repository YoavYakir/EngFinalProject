import pandas as pd
import numpy as np
import os

cwd = os.curdir()

# Load JSON files
json_path1 = f"{cwd}/../results/gtx2080/cpu/clean.json"
df = pd.read_json(json_path1)
json_path2 = f"{cwd}/../results/gtx2080/cpu/clean.json"
df2 = pd.read_json(json_path2)
json_path3 = f"{cwd}/../results/gtx2080/cpu/clean.json"
df3 = pd.read_json(json_path3)

# Concatenate the DataFrames
df = pd.concat([df, df2, df3], ignore_index=True)

# Print initial DataFrame and shape
print(df.head())
print(df.shape)

# Adjust column names for clarity
df['Run Type (CPU/GPU)'] = df['Run Type']

# Split the "Test Name" column into "Run Type" and "Test Name"
df[['Run Type', 'Test Name']] = df['Test Name'].str.split('_', 1, expand=True)  # Split only on the first underscore

# Verify unique test names
print(df['Test Name'].unique())

# Get unique test names for creating separate sheets
unique_tests = df['Test Name'].unique()

# Create an Excel writer to save each test in separate sheets
with pd.ExcelWriter(r"C:\Users\User\Downloads\cpu_tests_full.xlsx") as writer:
    for test_name in unique_tests:
        # Filter the DataFrame for each unique test name
        test_df = df[df['Test Name'] == test_name]
        
        # Write each filtered DataFrame to a separate sheet
        test_df.to_excel(writer, sheet_name=test_name, index=False)

# Output to confirm completion
print(f"New Excel file with separate sheets for each test type has been created.")