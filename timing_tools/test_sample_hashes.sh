#!/usr/bin/env bash

set -e

dir=$(dirname -- "$0")

tool="llvm-cycles"

arches=("haswell" "skylake" "ivybridge")
arch_prefixes=("hsw" "skl" "ivb")
arch_reg_mdl=("hsw.mdl" "skl_predictor.mdl" "ivb_predictor.mdl")
arch_reg_dump=("hsw.dump" "skl_predictor.dump" "ivb_predictor.dump")
arch_tsf_mdl=("hsw_tsf.mdl" "skl_tsf.mdl" "ivb_tsf.mdl")
arch_tsf_dump=("hsw_tsf.dump" "skl_tsf.dump" "ivb_tsf.dump")

# check all arrays have the same length
if [[ ${#arches[@]} -ne ${#arch_prefixes[@]} ]]; then
    echo "Error: arches and arch_prefixes arrays must have the same length."
    exit 1
fi
if [[ ${#arch_reg_mdl[@]} -ne ${#arch_reg_dump[@]} ]]; then
    echo "Error: arch_reg_mdl and arch_reg_dump arrays must have the same length."
    exit 1
fi
if [[ ${#arch_tsf_mdl[@]} -ne ${#arch_tsf_dump[@]} ]]; then
    echo "Error: arch_tsf_mdl and arch_tsf_dump arrays must have the same length."
    exit 1
fi

# foreach arch
for i in "${!arches[@]}"; do
    arch=${arches[i]}
    csv="${arch_prefixes[i]}.csv"
    arch_reg_mdl="${arch_reg_mdl[i]}"
    arch_reg_dump="${arch_reg_dump[i]}"
    arch_tsf_mdl="${arch_tsf_mdl[i]}"
    arch_tsf_dump="${arch_tsf_dump[i]}"
    echo "arch: $arch"
    echo "arch_csv: $csv"
    echo "arch_reg_mdl: $arch_reg_mdl"
    echo "arch_reg_dump: $arch_reg_dump"
    echo "arch_tsf_mdl: $arch_tsf_mdl"
    echo "arch_tsf_dump: $arch_tsf_dump"

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

    mkdir -p "${dir}/hash_csvs"

    # feed codeids from <arch_csv>.csv (first column, including empty lines) into test_code_hashes.py
    # cut -d, -f1 $csv | \
    # awk '{print length ? $0 : "\"\""}' | \
    # xargs -P4 -n 256 "${dir}/test_code_hash.py" $arch $tool >> "${dir}/hash_csvs/sampled_${tool}_${arch}.csv" 2>> /dev/null
    # 2>> "${dir}/hash_csvs/sampled_${tool}_${arch}.err"

    # Compare line counts between original CSV and output CSV
    orig_lines=$(wc -l < "$csv")
    out_lines=$(wc -l < "${dir}/hash_csvs/sampled_${tool}_${arch}.csv")
    echo "Original lines in $csv: $orig_lines"
    echo "Output lines in sampled_${tool}_${arch}.csv: $out_lines"
    echo "Difference: $((orig_lines - out_lines))"

    python ./learning/main.py --mode predict --block_csv "$csv" --predictor_file "$arch_reg_dump" --model_file "$arch_reg_mdl" \
    2>> /dev/null \
    | grep -oP '(?<=bt: ).*' >> "${dir}/hash_csvs/sampled_reg_${arch}.csv"
    # 2>> "${dir}/hash_csvs/sampled_reg_${arch}.err" \

    # python ./learning/main.py --mode predict --block_csv "$csv" --predictor_file "$arch_tsf_dump" --model_file "$arch_tsf_mdl" \
    # >> "${dir}/hash_csvs/sampled_tsf_${arch}.csv" \
    # 2>> "${dir}/hash_csvs/sampled_tsf_${arch}.err"
    # 2>> /dev/null

done