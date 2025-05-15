# Intro

This document details the steps needed to run and acquire data for the UIUC CS521 Final Project.

## Steps

Series of commands, while user/host might not be same, follow with current directory listing per cmd (use `cd` in between steps if needed).

Make sure docker is running on your machine. Use WSL(2) if possible (cuda passthru setup might be needed as well).

```bash
mohnish@DKUNMOH:~/source/repos$ git clone https://github.com/tninesling/ithemal.git
```

```bash
mohnish@DKUNMOH:~/source/repos$ git clone https://github.com/tninesling/bhive.git
mohnish@DKUNMOH:~/source/repos/bhive/benchmark/throughput$ cp -r --interactive . ../../../ithemal
```

```bash
mohnish@DKUNMOH:~/source/repos$ git clone https://github.com/tninesling/ithemal-models.git
mohnish@DKUNMOH:~/source/repos/ithemal-models/update-2025$ cp -r --interactive . ../../ithemal
```

```bash
mohnish@DKUNMOH:~/source/repos/ithemal$ source ./.venv/bin/activate
(.venv) mohnish@DKUNMOH:~/source/repos/ithemal/docker$ ./docker_build.sh
(.venv) mohnish@DKUNMOH:~/source/repos/ithemal/docker$ ./docker_connect.sh
(.venv) (ithemal) ithemal@0e75570f7e91:~/ithemal/timing_tools$ ./setup_tools.sh
(.venv) (ithemal) ithemal@0e75570f7e91:~/ithemal/data_collection/disassembler$ cmake -S . -B build/ -DLLVM_DIR=../../timing_tools/llvm-build/lib/cmake/llvm
(.venv) (ithemal) ithemal@0e75570f7e91:~/ithemal$ ./timing_tools/test_sample_hashes.sh
```
