#!/usr/bin/env bash

dir=$(dirname -- "$0")

tool="llvm-cycles"
# tool="llvm-rthroughput"

arches=("haswell" "skylake" "ivybridge")
arch_prefixes=("hsw" "skl" "ivb")
arch_reg_mdl=("hsw.mdl" "skl.mdl" "ivb.mdl")
arch_reg_dump=("hsw.dump" "skl.dump" "ivb.dump")
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

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -s           Enable sample creation (default)"
    echo "  -S           Disable sample creation"
    echo "  -h           Enable hash CSVs creation (default)"
    echo "  -H           Disable hash CSVs creation"
    echo "  -n SIZE      Set sample size (default: 10000)"
    echo "  -l           Enable llvm hash logic (default)"
    echo "  -L           Disable llvm hash logic"
    echo "  -r           Enable reg model logic (default)"
    echo "  -R           Disable reg model logic"
    echo "  -t           Enable tsf model logic (default)"
    echo "  -T           Disable tsf model logic"
    echo "  -g           Enable grouped CSV generation (default)"
    echo "  -G           Disable grouped CSV generation"
    echo "  -f           Enable figure generation (default)"
    echo "  -F           Disable figure generation"
    echo "  -?           Show this help message"
    exit 1
}

# param defaults here
CREATE_SAMPLES=1
CREATE_HASH_CSVS=1
SAMPLE_SIZE=10000
RUN_LLVM=1
RUN_REG=1
RUN_TSF=1
RUN_GROUPED_CSV=1
RUN_FIGURE_GEN=1

while getopts "sShHn:lLrRtTgGfF?" opt; do
    case $opt in
        s)
            CREATE_SAMPLES=1
            ;;
        S)
            CREATE_SAMPLES=0
            ;;
        h)
            CREATE_HASH_CSVS=1
            ;;
        H)
            CREATE_HASH_CSVS=0
            ;;
        n)
            SAMPLE_SIZE="$OPTARG"
            ;;
        l)
            RUN_LLVM=1
            ;;
        L)
            RUN_LLVM=0
            ;;
        r)
            RUN_REG=1
            ;;
        R)
            RUN_REG=0
            ;;
        t)
            RUN_TSF=1
            ;;
        T)
            RUN_TSF=0
            ;;
        g)
            RUN_GROUPED_CSV=1
            ;;
        G)
            RUN_GROUPED_CSV=0
            ;;
        f)
            RUN_FIGURE_GEN=1
            ;;
        F)
            RUN_FIGURE_GEN=0
            ;;
        ?)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

mkdir -p "${dir}/hash_csvs"
mkdir -p "${dir}/timing_output"

if [[ $CREATE_HASH_CSVS -eq 1 ]]; then
    # foreach arch
    for i in "${!arches[@]}"; do
        arch=${arches[i]}
        csv="${arch_prefixes[i]}.csv"
        csv_sample="${arch_prefixes[i]}_sample.csv"
        arch_reg_mdl="${arch_reg_mdl[i]}"
        arch_reg_dump="${arch_reg_dump[i]}"
        arch_tsf_mdl="${arch_tsf_mdl[i]}"
        arch_tsf_dump="${arch_tsf_dump[i]}"
        echo "arch: $arch"
        echo "arch_csv: $csv"
        echo "arch_csv_sample: $csv_sample"
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

        if [[ $CREATE_SAMPLES -eq 1 ]]; then
            # Take random N samples from the CSV file, excluding the empty hex
            shuf "$csv" | grep -v "^,.*$" | head -n "$SAMPLE_SIZE" > "$csv_sample"
        fi

        if [[ $RUN_LLVM -eq 1 ]]; then
            # feed codeids from <arch_csv>.csv (first column) into test_code_hashes.py
            cut -d, -f1 $csv_sample | \
            xargs -P4 -n 256 "${dir}/test_code_hash.py" $arch $tool >> "${dir}/hash_csvs/sampled_${tool}_${arch}.csv" \
            2>> /dev/null
        fi

        # 2>> "${dir}/hash_csvs/sampled_${tool}_${arch}.err"

        # Compare line counts between original CSV and output CSV
        orig_lines=$(wc -l < "$csv_sample")
        out_lines=$(wc -l < "${dir}/hash_csvs/sampled_${tool}_${arch}.csv")
        echo "Original lines in $csv_sample: $orig_lines"
        echo "Output lines in sampled_${tool}_${arch}.csv: $out_lines"
        echo "Difference: $((orig_lines - out_lines))"

        if [[ $RUN_REG -eq 1 ]]; then
            python ./learning/main.py --mode predict --block_csv "$csv_sample" --predictor_file "$arch_reg_dump" --model_file "$arch_reg_mdl" \
            2>> /dev/null \
            | grep -oP "(?<=bt: ).*" >> "${dir}/hash_csvs/sampled_reg_${arch}.csv"
            # >> "${dir}/hash_csvs/sampled_reg_${arch}.csv" \
            # 2>> "${dir}/hash_csvs/sampled_reg_${arch}.err"
        fi

        if [[ $RUN_TSF -eq 1 ]]; then
            python ./learning/transformer.py --mode predict --block_csv "$csv_sample" --predictor_file "$arch_tsf_dump" --model_file "$arch_tsf_mdl" \
            2>> /dev/null \
            | grep -oP "(?<=bt: ).*" >> "${dir}/hash_csvs/sampled_tsf_${arch}.csv"
            # >> "${dir}/hash_csvs/sampled_tsf_${arch}.csv" \
            # 2>> "${dir}/hash_csvs/sampled_tsf_${arch}.err"
        fi

    done
fi

if [[ $RUN_GROUPED_CSV -eq 1 ]]; then
    python "${dir}/generate_combined_csv.py"
fi
if [[ $RUN_FIGURE_GEN -eq 1 ]]; then
    python "${dir}/plot_grouped_cycles.py"
fi
