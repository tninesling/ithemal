#!/usr/bin/env bash

dir=$(dirname -- "$0")

tool="llvm-cycles"

arches=("haswell" "skylake" "ivybridge")
arch_csvs=("hsw.csv" "skl.csv" "ivb.csv")

if [[ ${#arches[@]} -ne ${#arch_csvs[@]} ]]; then
    echo "Error: arches and arch_csvs arrays must have the same length."
    exit 1
fi

# foreach arch
for i in "${!arches[@]}"; do
    echo "arch: ${arches[i]}"
    echo "arch_csv: ${arch_csvs[i]}"
    arch=${arches[i]}
    csv=${arch_csvs[i]}

    if [[ $arch = "haswell" ]]; then
        arch_id=1
    elif [[ $arch = "skylake" ]]; then
        arch_id=2
    elif [[ $arch = "broadwell" ]]; then
        arch_id=3
    elif [[ $arch = "nehalem" ]]; then
        arch_id=4
    elif [[ $arch = "ivybridge" ]]; then
        arch_id=5
    else
        echo "unknown arch $arch"
        exit 1
    fi

    mkdir -p "${dir}/build"

    # feed codeids from <arch_csv>.csv (first column, including empty lines) into test_code_hashes.py
    cut -d, -f1 $csv | \
    awk '{print length ? $0 : "\"\""}' | \
    xargs -P4 -n 256 "${dir}/test_code_hash.py" $arch $tool >> "${dir}/build/sampled_${tool}_${arch}.csv" 2>> /dev/null
    # 2>> "${dir}/build/sampled_${tool}_${arch}.err"

    # Compare line counts between original CSV and output CSV
    orig_lines=$(wc -l < "$csv")
    out_lines=$(wc -l < "${dir}/build/sampled_${tool}_${arch}.csv")
    echo "Original lines in $csv: $orig_lines"
    echo "Output lines in sampled_${tool}_${arch}.csv: $out_lines"
    echo "Difference: $((orig_lines - out_lines))"
done