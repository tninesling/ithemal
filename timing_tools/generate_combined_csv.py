import os
import csv
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
hash_csvs_dir = os.path.join(script_dir, "hash_csvs")
timing_output_dir = os.path.join(script_dir, "timing_output")
output_file = os.path.join(timing_output_dir, "sampled_combined_hashes.csv")
grouped_output_file = os.path.join(timing_output_dir, "sampled_combined_hashes_grouped.csv")
accuracy_csv = os.path.join(timing_output_dir, 'tool_accuracy_table.csv')

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

# add *_sample.csv files as ithemal-og, mapping arch
arch_map = {"hsw": "haswell", "skl": "skylake", "ivb": "ivybridge"}
root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
for sample_csv in os.listdir(root_dir):
    if not sample_csv.endswith('_sample.csv'):
        continue
    print(f"Processing {sample_csv}")
    prefix = sample_csv.split('_')[0]
    arch = arch_map.get(prefix, prefix)
    with open(os.path.join(root_dir, sample_csv), newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            code_hash, cycles = row[0], row[1]
            rows.append([code_hash, "ithemal-og", arch, cycles])

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

# non og cols
cols = [col for col in pivot_df.columns if col.endswith('_cycles') and col != 'ithemal-og_cycles']
archs = sorted(pivot_df['arch'].unique())
accuracy_rows = []
for arch in archs:
    arch_df = pivot_df[pivot_df['arch'] == arch]
    row = [arch]
    truth = arch_df['ithemal-og_cycles'].astype(float)
    for col in cols:
        local = arch_df[col].astype(float)
        mask = (~truth.isna()) & (~local.isna())
        # just in case nan math
        if mask.sum() == 0:
            acc = float('nan')
        else:
            acc = ((abs(local[mask] - truth[mask]) <= 0.25 * truth[mask]).sum() / mask.sum())
        row.append(acc)
    accuracy_rows.append(row)
# make an append the overall row
row = ['overall']
truth = pivot_df['ithemal-og_cycles'].astype(float)
for col in cols:
    local = pivot_df[col].astype(float)
    mask = (~truth.isna()) & (~local.isna())
    if mask.sum() == 0:
        acc = float('nan')
    else:
        acc = ((abs(local[mask] - truth[mask]) <= 0.25 * truth[mask]).sum() / mask.sum())
    row.append(acc)
accuracy_rows.append(row)
accuracy_header = ['arch'] + [col.replace('_cycles', '') for col in cols]
accuracy_df = pd.DataFrame(accuracy_rows, columns=accuracy_header)
accuracy_df.to_csv(accuracy_csv, index=False)
print(f"Accuracy table written to {accuracy_csv}")
