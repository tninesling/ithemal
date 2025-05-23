# Intro

This document details the steps needed to run and acquire data for the UIUC CS521 Final Project.

## Steps

Series of commands, while user/host might not be same, follow with current directory listing per cmd (use `cd` in between steps if needed).

Make sure docker is running on your machine. Use WSL(2) if possible (cuda passthru setup might be needed as well). Email both of us if any issues arise with machine config/setup.

```bash
mohnish@DKUNMOH:~/source/repos$ git clone https://github.com/tninesling/ithemal.git
```

- NOTE: above cmd may already be done, just showing where ithemal repo is relative to other cmds below

```bash
mohnish@DKUNMOH:~/source/repos$ git clone https://github.com/tninesling/bhive.git
mohnish@DKUNMOH:~/source/repos/bhive/benchmark/throughput$ cp -r --interactive . ../../../ithemal
```

- gets the hsw.csv, etc. files into ithemal root

```bash
mohnish@DKUNMOH:~/source/repos$ git clone https://github.com/tninesling/ithemal-models.git
mohnish@DKUNMOH:~/source/repos/ithemal-models/update-2025$ cp -r --interactive . ../../ithemal
```

- gets the pre-dumped models into ithemal root

### Functional Steps

As follows is each step to run thru and get data. Make sure to read the comments below each step before executing to see if it is needed.

```bash
mohnish@DKUNMOH:~/source/repos/ithemal/docker$ ./docker_build.sh
```

- NOTE: takes ~20m

```bash
mohnish@DKUNMOH:~/source/repos/ithemal/docker$ ./docker_connect.sh
```

- just connects up, putting you into the terminal of the docker container

```bash
(ithemal) ithemal@0e75570f7e91:~$ cd ithemal/timing_tools
(ithemal) ithemal@0e75570f7e91:~/ithemal/timing_tools$ ./setup_tools.sh
```

- build llvm mainly ~5m

```bash
(ithemal) ithemal@0e75570f7e91:~/ithemal/timing_tools$ cd ~/ithemal/data_collection/disassembler
(ithemal) ithemal@0e75570f7e91:~/ithemal/data_collection/disassembler$ cmake -S . -B build/ -DLLVM_DIR=../../timing_tools/llvm-build/lib/cmake/llvm
```

- cmake the disasm tool (after getting llvm dir)

```bash
(ithemal) ithemal@0e75570f7e91:~/ithemal/data_collection/disassembler$ cd build
(ithemal) ithemal@0e75570f7e91:~/ithemal/data_collection/disassembler/build$ make
```

- actually build the disasm tool (after getting llvm dir)

```bash
(ithemal) ithemal@0e75570f7e91:~/ithemal/data_collection/disassembler/build$ cd ~/ithemal
(ithemal) ithemal@0e75570f7e91:~/ithemal$ ./timing_tools/test_sample_hashes.sh
```

- ~20m, grabs a 10k sample (param to configure), saves out to `<arch>_sample.csv`, then runs llvm-mca, regular, transfomer on sample: combine with ithemal og data, present via graphs
- throws intermediate csvs into `timing_tools/hash_csvs/`, and _the final outputs_ (grouped csvs, figures/plots) in `timing_tools/timing_output/`
- NOTE: `./timing_tools/test_sample_hashes.sh -?` prints the fine grain control of each operation, allows for resuming in place of any errors (or reruns without a full recompute)
- NOTE: should remove files in `timing_tools/hash_csvs` as needed, since all operations concatenate to the csvs
- Already have pushed up to the repo our 3 arch sample csvs and their associated results, but running the above will regenerate a new sample -> new output.

#### Examples of test_sample_hashes.sh (check usage in script)

```bash
(ithemal) ithemal@0e75570f7e91:~/ithemal$ ./timing_tools/test_sample_hashes.sh -SH
```

- remake grouped csvs and plots only (**RECOMMENDED: Uses data already present without recomputation**)

```bash
(ithemal) ithemal@0e75570f7e91:~/ithemal$ ./timing_tools/test_sample_hashes.sh -SRT
```

- remake only llvm-mca cycles csv (remember to delete `timing_tools/hash_csvs/` files as needed)

```bash
(ithemal) ithemal@0e75570f7e91:~/ithemal$ ./timing_tools/test_sample_hashes.sh -SL
```

- remake only models (regular + transformer) csvs (remember to delete `timing_tools/hash_csvs/` files as needed)

```bash
(ithemal) ithemal@0e75570f7e91:~/ithemal$ ./timing_tools/test_sample_hashes.sh -n 142
```

- remake everything with reduced sample size (remember to delete `timing_tools/hash_csvs/` due to append only)

## Outputs

Plots, csvs in `timing_tools/timing_output/`
