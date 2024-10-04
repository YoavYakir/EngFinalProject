import pandas as pd
json_path = "EngFinalProject/results/results.json"
df = pd.read_json(json_path)
print(df.head())
df.to_excel("results_gpu.xlsx")
