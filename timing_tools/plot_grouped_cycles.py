import pandas as pd
import matplotlib.pyplot as plt
import os
from itertools import combinations

script_dir = os.path.dirname(os.path.abspath(__file__))
timing_output_dir = os.path.join(script_dir, "timing_output")
grouped_csv = os.path.join(timing_output_dir, "sampled_combined_hashes_grouped.csv")
df = pd.read_csv(grouped_csv)

# get cols and to_numeric them just in case they are treated as strings
tool_cols = [col for col in df.columns if col.endswith('_cycles')]
for col in tool_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# histo, cycles per tool
for col in tool_cols:
    plt.figure(figsize=(8, 6))
    plt.hist(df[col].dropna(), bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f"Distribution of {col}")
    plt.xlabel("Cycles")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(timing_output_dir, f"{col}_hist.png"))
    plt.close()

# scatter, pairwise of tools
for col1, col2 in combinations(tool_cols, 2):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[col1], df[col2], alpha=0.5, color='green', edgecolors='w', s=20)
    plt.title(f"{col1} vs {col2}")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.tight_layout()
    plt.savefig(os.path.join(timing_output_dir, f"{col1}_vs_{col2}_scatter.png"))
    plt.close()

# boxplot, cycles per tool, arch grouped
archs = df['arch'].unique()
for col in tool_cols:
    plt.figure(figsize=(8, 6))
    data = [df[df['arch'] == arch][col].dropna() for arch in archs]
    plt.boxplot(data, labels=archs)
    plt.title(f"{col} by Architecture")
    plt.xlabel("Architecture")
    plt.ylabel("Cycles")
    plt.tight_layout()
    plt.savefig(os.path.join(timing_output_dir, f"{col}_by_arch_boxplot.png"))
    plt.close()

print("Plots saved in current dir.")
