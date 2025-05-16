import os
import csv
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
hash_csvs_dir = os.path.join(script_dir, "hash_csvs")
timing_output_dir = os.path.join(script_dir, "timing_output")
output_file = os.path.join(timing_output_dir, "sampled_combined_hashes.csv")
grouped_output_file = os.path.join(timing_output_dir, "sampled_combined_hashes_grouped.csv")

header = ["code_hash", "tool", "arch", "cycles"]
rows = []

for csv_name in os.listdir(hash_csvs_dir):
    if not csv_name.startswith("sampled_") or not csv_name.endswith(".csv"):
        continue
    print(f"Processing {csv_name}")
    parts = csv_name.replace(".csv", "").split("_")
    if len(parts) < 3:
        continue
    tool = parts[1]
    arch = parts[2]
    with open(os.path.join(hash_csvs_dir, csv_name), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            code_hash, cycles = row[0], row[1]
            rows.append([code_hash, tool, arch, cycles])

combined_df = pd.DataFrame(rows, columns=header)
combined_df.to_csv(output_file, index=False)
print(f"Combined CSV written to {output_file}")

# group by arch,code_hash, with cycles per tool
pivot_df = combined_df.pivot_table(index=["arch", "code_hash"], columns=["tool"], values="cycles", aggfunc="first")
# flatten cols
pivot_df.columns = [f"{tool}_cycles" for tool in pivot_df.columns]
pivot_df.reset_index(inplace=True)
pivot_df.to_csv(grouped_output_file, index=False)
print(f"Grouped CSV written to {grouped_output_file}")