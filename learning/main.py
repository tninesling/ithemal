from contextlib import contextmanager
import os
import pandas as pd
import subprocess
import time
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import xml.etree.ElementTree as ET
import warnings

import common_libs.utilities as ut
from pytorch.data.data_cost import DataInstructionEmbedding, DataItem
from pytorch.ithemal import ithemal_utils

_TOKENIZER = os.path.join(
    os.environ["ITHEMAL_HOME"], "data_collection", "build", "bin", "tokenizer"
)
_fake_intel = "\n" * 500


def datum_of_code(data, block_hex):
    xml = subprocess.check_output([_TOKENIZER, block_hex, "--token"])
    data.raw_data = [(-1, -1, _fake_intel, xml)]
    data.data = []
    data.prepare_data(fixed=False, progress=False)
    return data.data[-1]


class BasicBlockCSV(Dataset):
    def __init__(self, csv_file, embedder):
        df = pd.read_csv(
            csv_file,
            header=None,
            sep=r"\s*,\s*",
            engine="python",
            names=["hex", "throughput"],
        )

        def is_valid_hex(s):
            # Check that s is a string and fits hexadecimal format
            if not isinstance(s, str):
                return False
            try:
                int(s, 16)
                return True
            except ValueError:
                return False

        # Filter the DataFrame to only include rows with valid hex strings
        df = df[df["hex"].apply(is_valid_hex)]

        self.blocks = df
        self.embedder = embedder

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        hex, throughput = self.blocks.iloc[idx, :]
        datum = datum_of_code(self.embedder, hex)
        return {"datum": datum, "throughput": throughput}

    @staticmethod
    def collate_fn(batch):
        data = [item["datum"] for item in batch]
        throughputs = torch.tensor(
            [item["throughput"] for item in batch], dtype=torch.float32
        )
        return {"datum": data, "throughput": throughputs}


def load_model_and_data(model_file, model_data_file):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", torch.serialization.SourceChangeWarning)
        (model, data) = ithemal_utils.load_model_and_data(model_file)

    state_dict = torch.load(model_data_file)
    model_dict = model.state_dict()
    new_model_dict = {
        k: v for (k, v) in list(state_dict["model"].items()) if k in model_dict
    }
    model_dict.update(new_model_dict)
    model.load_state_dict(model_dict)

    return (model, data)


def save_checkpoint(model, optimizer, epoch, batch_num, filename, **rest):
    state_dict = {
        "epoch": epoch,
        "batch_num": batch_num,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    for k, v in list(rest.items()):
        state_dict[k] = v

    try:
        os.makedirs(os.path.dirname(filename))
    except OSError:
        pass

    torch.save(state_dict, filename)


def load_block_csv(block_csv, embedding):
    blocks = os.path.join(os.environ["ITHEMAL_HOME"], block_csv)
    if not os.path.exists(blocks):
        raise FileNotFoundError(
            f"File {blocks} does not exist. Please ensure the path is correct."
        )
    dataset = BasicBlockCSV(blocks, embedding)

    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # rest for testing
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=BasicBlockCSV.collate_fn,  # custom collate allows returning objects that aren't tensors (we need DataItem)
        num_workers=16,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=BasicBlockCSV.collate_fn,
        num_workers=8,
        pin_memory=True,
    )

    return (train_dataloader, test_dataloader)


def normalized_mse_loss(output, target):
    loss = torch.nn.functional.mse_loss(output, target, reduction="none")
    loss = torch.sqrt(loss) / (target + 1e-3)
    return loss.mean()


def train(block_csv, predictor_file, model_file, num_epochs=100, tolerance=10):
    embedding = DataInstructionEmbedding()
    (train_dataloader, _test_dataloader) = load_block_csv(block_csv, embedding)
    batches_per_epoch = len(train_dataloader) // num_epochs

    embed_file = os.path.join(
        os.environ["ITHEMAL_HOME"],
        "learning",
        "pytorch",
        "inputs",
        "embeddings",
        "code_delim.emb",
    )
    if not os.path.exists(embed_file):
        raise FileNotFoundError(
            f"Embedding file {embed_file} does not exist. Please ensure the path is correct."
        )

    # Params used by training invocation script from README
    params = ithemal_utils.BaseParameters(
        data="hsw.csv",
        embed_mode="none",
        embed_file=embed_file,  # TODO: How do we load that into DataInstructionEmbedding?
        random_edge_freq=0.0,
        predict_log=False,
        no_residual=False,
        no_dag_rnn=False,
        dag_reduction=ithemal_utils.md.ReductionType.MAX,
        edge_ablation_types=[],
        embed_size=256,
        hidden_size=256,
        linear_embeddings=False,
        use_rnn=True,
        rnn_type=ithemal_utils.md.RnnType.LSTM,
        rnn_hierarchy_type=ithemal_utils.md.RnnHierarchyType.MULTISCALE,
        rnn_connect_tokens=False,
        rnn_skip_connections=False,
        rnn_learn_init=False,
        no_mem=False,
        linear_dependencies=False,
        flat_dependencies=False,
        dag_nonlinearity=None,
        dag_nonlinearity_width=128,
        dag_nonlinear_before_max=False,
    )

    model = ithemal_utils.load_model(params, embedding)

    # model = ithemal_utils.md.RNN(rnn_params)
    # model.set_learnable_embedding(mode=params.embed_mode, dictsize=628)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i, batch in enumerate(train_dataloader):
        data, throughputs = batch["datum"], batch["throughput"].to(device)
        optimizer.zero_grad()

        batch_correct = torch.tensor(0.0, device=device)
        batch_loss = torch.tensor(0.0, device=device)
        for datum, throughput in zip(data, throughputs):
            # Each datum is a block with a list of instructions
            output = model(datum)
            loss = normalized_mse_loss(output, throughput)
            batch_loss += loss
            loss.backward()
            optimizer.step()

            diff = output - throughput
            is_correct = torch.abs(diff * 100 / (throughput + 1e-3)) < tolerance
            batch_correct += torch.sum(is_correct).item()

        if i % batches_per_epoch == 0:
            print(
                f"Epoch {i // batches_per_epoch} (Batch {i}/{len(train_dataloader)}) | Loss: {batch_loss.item():.4f}, Batch Accuracy: {batch_correct.item() / len(data) * 100:.2f}%"
            )

    print("Training complete.")
    ithemal_utils.dump_model_and_data(model, embedding, predictor_file)
    save_checkpoint(model, optimizer, epoch="final", batch_num=0, filename=model_file)
    print("Model and data saved successfully.")


def test(block_csv, predictor_file, model_file, num_epochs=100, tolerance=10):
    (model, embedding) = load_model_and_data(predictor_file, model_file)
    (_train_dataloader, test_dataloader) = load_block_csv(block_csv, embedding)
    batches_per_epoch = len(test_dataloader) // num_epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("Starting evaluation...")
    with torch.no_grad():
        correct = torch.tensor(0.0, device=device)
        total = torch.tensor(0.0, device=device)
        for i, batch in enumerate(test_dataloader):
            data, throughputs = batch["datum"], batch["throughput"].to(device)
            for datum, throughput in zip(data, throughputs):
                output = model(datum)

                diff = output - throughput
                is_correct = torch.abs(diff * 100 / (throughput + 1e-3)) < tolerance
                correct += torch.sum(is_correct).item()
                total += 1

            if i % batches_per_epoch == 0:
                accuracy = correct.item() / total.item() if total.item() > 0 else 0
                print(
                    f"Epoch {i // batches_per_epoch} (Batch {i}/{len(test_dataloader)}) | Accuracy: {accuracy * 100:.2f}%"
                )

        accuracy = correct.item() / total.item() if total.item() > 0 else 0
        print(f"Final Accuracy: {accuracy * 100:.2f}%")


def predict(block_hex, predictor_file, model_file):
    (model, embedding) = load_model_and_data(predictor_file, model_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    datum = datum_of_code(embedding, block_hex)
    with torch.no_grad():
        output = model(datum)
        print(f"Predicted throughput for block {block_hex}: {output.item():.4f}")


@contextmanager
def timer():
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    block_csv = "hsw.csv"
    predictor_file = "hsw_predictor.dump"
    model_file = "hsw_predictor.mdl"
    tolerance = 25

    print(f"Config: {block_csv}, {predictor_file}, {model_file}, tolerance={tolerance}")

    print("Begin training...")
    with timer():
        train(
            block_csv,
            predictor_file,
            model_file,
            num_epochs=20,
            tolerance=tolerance,
        )

    print("Begin testing...")
    with timer():
        test(
            block_csv,
            predictor_file,
            model_file,
            num_epochs=10,
            tolerance=tolerance,
        )

    """
    print("Begin prediction...")
    with timer():
        predict(
            block_hex="4183ff0119c083e00885c98945c4b8010000000f4fc139c2",
            predictor_file="skl_predictor.dump",
            model_file="skl_predictor.mdl",
        )
    """
