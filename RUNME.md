# Intro

This document details the steps needed to run and acquire data for the UIUC CS521 Final Project.

## Steps

Series of commands, while user/host might not be same, follow with current directory listing per cmd (use `cd` in between steps if needed).

Make sure docker is running on your machine. Use WSL(2) if possible (cuda passthru setup might be needed as well).

```bash
mohnish@DKUNMOH:~/source/repos$ git clone https://github.com/tninesling/ithemal.git
```

NOTE: above cmd may already be done, just showing where ithemal is relative to other cmds below

```bash
mohnish@DKUNMOH:~/source/repos$ git clone https://github.com/tninesling/bhive.git
mohnish@DKUNMOH:~/source/repos/bhive/benchmark/throughput$ cp -r --interactive . ../../../ithemal
```

```bash
mohnish@DKUNMOH:~/source/repos$ git clone https://github.com/tninesling/ithemal-models.git
mohnish@DKUNMOH:~/source/repos/ithemal-models/update-2025$ cp -r --interactive . ../../ithemal
```

### Functional Steps

As follows is each step to run thru and get data.

```bash
mohnish@DKUNMOH:~/source/repos/ithemal/docker$ ./docker_build.sh
```

- NOTE: takes ~20m

```bash
mohnish@DKUNMOH:~/source/repos/ithemal/docker$ ./docker_connect.sh
```

- just connects up, putting you into the terminal of the docker container

```bash
(ithemal) ithemal@0e75570f7e91:~/ithemal/timing_tools$ ./setup_tools.sh
```

- build llvm mainly ~5m

```bash
(ithemal) ithemal@0e75570f7e91:~/ithemal/data_collection/disassembler$ cmake -S . -B build/ -DLLVM_DIR=../../timing_tools/llvm-build/lib/cmake/llvm
```

- build the disasm tool (after getting llvm dir)

```bash
(ithemal) ithemal@0e75570f7e91:~/ithemal$ conda install gevent
```

- the `timing_tools/test_code_hash.py` script needs gevent but apparently wasnt included in the conda env by default
- just add this if no records get genned for llvm-cycles tool (usually on every docker startup)

```bash
(ithemal) ithemal@0e75570f7e91:~/ithemal$ ./timing_tools/test_sample_hashes.sh
```

- ~20m, grabs a 10k sample (param to configure), saves out to `<arch>_sample.csv`, then runs llvm-mca, regular, transfomer on sample: combine with ithemal og data, present via graphs
- NOTE: `./timing_tools/test_sample_hashes.sh -?` prints the fine grain control of each operation, allows for resuming in place of any errors (or reruns without a full recompute)
