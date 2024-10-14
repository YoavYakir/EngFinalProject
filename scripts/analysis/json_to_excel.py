import pandas as pd
json_path = "results/results.json"
df = pd.read_json(json_path)
print(df.head())
df.to_excel("results/results_dudu_231024.xlsx")
