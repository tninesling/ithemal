
# Overview

Ithemal is a data driven model for predicting throughput of a basic block of x86-64 instructions.

More details about Ithemal's approach can be found in our paper.

* [Ithemal: Accurate, Portable and Fast Basic Block Throughput Estimation using Deep Neural Networks](https://arxiv.org/abs/1808.07412)<br>
   Charith Mendis, Alex Renda, Saman Amarasinghe, Michael Carbin<br>
   Proceedings of the 36th International Conference on Machine Learning (ICML) 2019.
   [Bibtex](http://groups.csail.mit.edu/commit/bibtex.cgi?key=ithemal-icml)

# Usage

## Environment Setup

You first need to install [docker](https://github.com/docker/docker-install) and [docker-compose](https://docs.docker.com/compose/install/).

It is easiest to run Ithemal within the provided docker environment.
To build the docker environment, run `docker/docker_build.sh`.
No user interaction is required during the build, despite the various prompts seemingly asking for input.

Once the docker environment is built, connect to it with `docker/docker_connect.sh`.
This will drop you into a tmux shell in the container.
It will also start a Jupyter notebook exposed on port 8888 with the password `ithemal`; nothing depends on this except for your own convenience, so feel free to disable exposing the notebook by removing the port forwarding on lines 37 and 38 in `docker/docker-compose.yml`.
The file system in the container is mounted from the local file system, so changes to the file system on the host will propagate to the docker instance, and vice versa.
The container will continue running in the background, even if you exit.
The container can be stopped with `docker/docker_stop.sh` from the host machine.
To detach from the container while keeping jobs running, use the normal tmux detach command of `Control-b d`; running `docker/docker_connect.sh` will drop you back into the same session.

## Prediction

Models can be downloaded from [the Ithemal models repository](https://www.github.com/psg-mit/Ithemal-models).
Models are split into two parts: the model architecture and the model data (this is an unfortunate historical artifact more than a good design decision).
The model architecture contains the code and the token embedding map for the model, and the model data contains the learned model tensors.
The versions of the model reported in the paper are:
- The [Haswell model](https://github.com/psg-mit/Ithemal-models/blob/master/paper/haswell/)
- The [Skylake model](https://github.com/psg-mit/Ithemal-models/blob/master/paper/skylake/)
- The [Ivy Bridge model](https://github.com/psg-mit/Ithemal-models/blob/master/paper/ivybridge/)

### Command Line

Ithemal can be used as a drop-in replacement for the throughput prediction capabilities of `IACA` or `llvm-mca` via the `learning/pytorch/ithemal/predict.py` script.
Ithemal uses the same convention as `IACA` to denote what code should be predicted; this can be achieved with any of the files in `learning/pytorch/examples`, or by consulting [the IACA documentation](https://software.intel.com/en-us/articles/intel-architecture-code-analyzer).

Once you have downloaded one of the models from the previous section, and you have compiled a piece of code to some file, you can generate prediction for this code with something like:
```
python learning/pytorch/ithemal/predict.py --verbose --model predictor.dump --model-data trained.mdl --file a.out
```

The predictor can also be run with a hex-encoded version of a block with the `--raw-stdin` flag. For example, run an example block from [Bhive](https://github.com/ithemal/bhive/blob/master/benchmark/throughput/hsw.csv) as:
```bash
echo 4183ff0119c083e00885c98945c4b8010000000f4fc139c2 | \
   python learning/pytorch/ithemal/predict.py --verbose --model predictor.dump --model-data trained.mdl --raw-stdin
```

## Training

### Data

#### Representation

Raw data is represented as a list of tuples containing `(code_id, timing, code_intel, code_xml)`, where `code_id` is a unique identifier for the code, `timing` is the float representing the timing of the block, `code_intel` is the newline-separated human-readable block string (tis is only for debugging and can be empty or `None`), and `code_xml` is the result of the tokenizer on that block (detailed in the [Canonicalization section](#canonicalization)).
To store datasets, we `torch.save` and `torch.load` this list of tuples.
The first 80% of a dataset is loaded as the train set, and the last 20% is the test set.
For an example of one of these datasets, look at [a small sample of our training data](https://github.com/psg-mit/Ithemal-models/blob/master/paper/data/haswell_sample1000.data).

#### Canonicalization

To canonicalize a basic block so that it can be used as input for Ithemal, first get the hex representation of the basic block you want to predict (i.e. via `xxd`, `objdump`, or equivalent).
For instance, the instruction `push   0x000000e3` is represented in hex as `68e3000000`.
Next, run the tokenizer as follows:
```
data_collection/build/bin/tokenizer {HEX_CODE} --token
```
which will output an XML representation of the basic block, with all implicit operands enumerated.

### Model Training

To train a model, pick a suitable `EXPERIMENT_NAME` and `EXPERIMENT_TIME`, and run the following command:
```
python learning/pytorch/ithemal/run_ithemal.py --data {DATA_FILE} --use-rnn train --experiment-name {EXPERIMENT_NAME} --experiment-time {EXPERIMENT_TIME} --sgd --threads 4 --trainers 6 --weird-lr --decay-lr --epochs 100
```
which will train a model with the parameters reported in the paper.
The results of this are saved into `learning/pytorch/saved/EXPERIMENT_NAME/EXPERIMENT_TIME/`, which we will refer to as `RESULT_DIR` from now on.
The training loss is printed live as the model is trained, and also saved into `RESULT_DIR/loss_report.log`, which is a tab-separated list of `epoch, elapsed time, training loss, number of active parallel trainers`.
The results of the trained model on the test set are stored in `RESULT_DIR/validation_results.txt`, which consists of a list of of the `predicted,actual` value of each item in the test set, followed by the overall loss of the trained model on the test set at the end.
Finally, the resulting trained models and predictor dumps (for use in the command line API above) are saved in `RESULT_DIR/trained.mdl` and `RESULT_DIR/predictor.dump` respectively.

# Code Structure

- The command-line and training infrastructure is in `learning/pytorch/ithemal/`.
- The models and optimizer are in `learning/pytorch/models/graph_models.py` and `learning/pytorch/models/train.py`
- The data infrastructure is primarily in `learning/pytorch/data/data_cost.py`
- The representation of instructions and basic blocks are in `common/common_libs/utilities.py`
- The canonicalization and tokenization are in `data_collection/tokenizer/tokenizer.c` and `data_collection/common/`

# Data Collection

This repo contains a preliminary version of our data collection infrastructure; we plan on releasing a more detailed description and infrastructure soon.

# 2025 Training Update

In order to better utilize available hardware, we have upgraded to a CUDA-enabled version of PyTorch.
This change requires reworking the training setup to get data to the GPU as quickly as possible.

Assuming you have the Bhive CSVs available locally, you can run the `learning/main.py` script to begin
training, followed by a test with the remaining 20% of the data. The data loaders run subprocesses to
execute the local tokenizer on each block's code, and the embedding canonicalizes the tokenized block.

Inside `learning/main.py`, follow params in the main block to use the bhive file you want.
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ithemal Learning Pipeline")
    parser.add_argument("--mode", choices=["train", "test", "predict"], required=True, help="Mode to run: train, test, or predict")
    parser.add_argument("--block_csv", type=str, default="hsw.csv", help="CSV file with blocks")
    parser.add_argument("--predictor_file", type=str, default="hsw_predictor.dump", help="Predictor dump file")
    parser.add_argument("--model_file", type=str, default="hsw_predictor.mdl", help="Model file")
    parser.add_argument("--tokenized_blocks_file", type=str, default="hsw_tokenized_blocks.pkl", help="Tokenized blocks pickle file")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs for training")
    parser.add_argument("--tolerance", type=int, default=25, help="Tolerance for accuracy calculation")
    args = parser.parse_args()

    # ...
```

Same is true for `learning/transformer.py`

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transformer Ithemal Pipeline")
    parser.add_argument("--mode", choices=["predict"], required=True, help="Mode to run: predict only for now")
    parser.add_argument("--block_csv", type=str, default="hsw.csv", help="CSV file with blocks")
    parser.add_argument("--predictor_file", type=str, default="hsw_tsf_predictor.pkl", help="Predictor dump file")
    parser.add_argument("--model_file", type=str, default="hsw_tsf_model.pkl", help="Model file")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--tolerance", type=int, default=25, help="Tolerance for accuracy calculation")
    args = parser.parse_args()

    # ...
```

The predictor, model, and tokenized blocks files will be empty on the first run. Intermediate stages of the model will be saved during execution.

[our ithemal-models fork](https://github.com/tninesling/ithemal-models/tree/master/update-2025) has some premade model dumps if needed (e.g. used in below evaluation).

# 2025 Evaluation Update

To actually generate reports, analysis, etc from the evaluation, see the [RUNME.md](RUNME.md) file.

The current *_sample.csv files at the project root + files in the `timing_tools/timing_output` and `timing_tools/hash_csvs` are what were used in the evaluation, able to run as as with these files (again see RUNME.md above).
