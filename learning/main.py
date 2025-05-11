import concurrent
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import pandas as pd
import pickle
import subprocess
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt

from pytorch.data.data_cost import DataInstructionEmbedding
from pytorch.ithemal import ithemal_utils

_TOKENIZER = os.path.join(
    os.environ["ITHEMAL_HOME"], "data_collection", "build", "bin", "tokenizer"
)
_fake_intel = "\n" * 500


def tokenize_single(hex_str):
    return subprocess.check_output([_TOKENIZER, hex_str, "--token"])


def datum_of_code(data: DataInstructionEmbedding, block_hex):
    xml = tokenize_single(block_hex)
    data.raw_data = [(-1, -1, _fake_intel, xml)]
    data.data = []
    data.prepare_data(fixed=False, progress=False)
    return data.data[-1]


def is_valid_hex(s):
    # Check that s is a string and fits hexadecimal format
    if not isinstance(s, str):
        return False
    try:
        int(s, 16)
        return True
    except ValueError:
        return False


class BasicBlockCSV(Dataset):
    def __init__(self, csv_file, embedder):
        self.df = pd.read_csv(
            csv_file,
            header=None,
            sep=r"\s*,\s*",
            engine="python",
            names=["hex", "throughput"],
        )
        self.embedder = embedder

    def save(self, filepath):
        data_to_save = {
            "dataframe": self.df,
            "embedder_data": self.embedder.data,
            "embedder_raw_data": self.embedder.raw_data,
            "token_to_hot_idx": self.embedder.token_to_hot_idx,
            "hot_idx_to_token": self.embedder.hot_idx_to_token,
            "mem_start": self.embedder.mem_start,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data_to_save, f)

    @classmethod
    def load(cls, filepath, embedder=None):
        # Load saved data
        with open(filepath, "rb") as f:
            saved_data = pickle.load(f)

        if embedder is None:
            embedder = DataInstructionEmbedding()

        instance = cls.__new__(cls)
        instance.df = saved_data["dataframe"]
        instance.embedder = embedder

        instance.embedder.data = saved_data["embedder_data"]
        instance.embedder.raw_data = saved_data["embedder_raw_data"]
        instance.embedder.token_to_hot_idx = saved_data["token_to_hot_idx"]
        instance.embedder.hot_idx_to_token = saved_data["hot_idx_to_token"]
        instance.embedder.mem_start = saved_data["mem_start"]

        return instance

    """ Dataset related methods """

    def __len__(self):
        return len(self.embedder.data)

    def __getitem__(self, idx):
        throughput = self.df["throughput"].iloc[idx]
        block = self.embedder.data[idx]
        return {"block": block, "throughput": throughput}

    @staticmethod
    def collate_fn(batch):
        blocks = [item["block"] for item in batch]
        throughput = torch.tensor(
            [item["throughput"] for item in batch], dtype=torch.float32
        )
        return {"blocks": blocks, "throughput": throughput}

    """ Builder methods """

    def remove_invalid_hex(self):
        print("Removing invalid hex...")
        self.df = self.df[self.df["hex"].apply(is_valid_hex)]
        return self

    def tokenize_hex(self):
        print("Tokenizing hex...")

        # Determine number of workers - use max(1, available_cores - 1) to avoid overloading the system
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        print(f"Using {num_workers} workers for tokenization.")

        # Create list to store results in the correct order
        results = [None] * len(self.df)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks and get futures
            hex_values = self.df["hex"].tolist()
            futures = {
                executor.submit(tokenize_single, hex_val): i
                for i, hex_val in enumerate(hex_values)
            }

            # Process results as they complete with progress bar
            with tqdm(total=len(futures), desc="Tokenizing") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        print(f"Error tokenizing entry {idx}: {e}")
                        # Set a default value or handle the error appropriately
                        results[idx] = None
                    pbar.update(1)

        # Update the dataframe with tokenized results
        self.df["tokenized"] = results
        return self

    def load_into_embedder(self):
        print("Loading into embedder...")
        self.embedder.raw_data = [
            (-1, -1, _fake_intel, x) for x in self.df["tokenized"].tolist()
        ]
        return self

    def process_into_blocks(self):
        print("Processing into blocks...")
        self.embedder.data = []
        self.embedder.prepare_data(fixed=False, progress=True)
        return self

    def create_loaders(self):
        train_size = int(0.8 * len(self))  # 80% for training
        test_size = len(self) - train_size  # rest for testing
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            collate_fn=BasicBlockCSV.collate_fn,  # custom collate allows returning non-tensors from batch
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


def load_block_csv(block_csv, embedding, tokenized_blocks_file):
    blocks = os.path.join(os.environ["ITHEMAL_HOME"], block_csv)
    if not os.path.exists(blocks):
        raise FileNotFoundError(
            f"File {blocks} does not exist. Please ensure the path is correct."
        )

    dataset = None
    if os.path.exists(tokenized_blocks_file):
        print(f"Loading dataset from {tokenized_blocks_file}...")
        dataset = BasicBlockCSV.load(tokenized_blocks_file, embedder=embedding)
    else:
        dataset = BasicBlockCSV(blocks, embedding)
        dataset.remove_invalid_hex()
        dataset.tokenize_hex()
        dataset.load_into_embedder()
        dataset.process_into_blocks()
        dataset.save(tokenized_blocks_file)
        print(f"Dataset saved to {tokenized_blocks_file}.")

    return dataset.create_loaders()


def normalized_mse_loss(output, target):
    loss = torch.nn.functional.mse_loss(output, target, reduction="none")
    loss = torch.sqrt(loss) / (target + 1e-3)
    return loss.mean()

def check_correct(outputs, throughput, tolerance=25):
    return torch.abs((outputs - throughput) * 100 / (throughput + 1e-3)) < tolerance

def plot(accuracies, losses):
    _fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    axs[0].plot(accuracies)
    axs[0].set_title("Accuracy")
    axs[0].set_xlabel("Batch")
    axs[0].set_ylabel("Accuracy (%)")

    axs[1].plot(losses)
    axs[1].set_title("Loss")
    axs[1].set_xlabel("Batch")
    axs[1].set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig("training_metrics.png")

def train(block_csv, predictor_file, model_file, tokenized_blocks_file, num_epochs, tolerance):
    embedding = DataInstructionEmbedding()
    (train_dataloader, _test_dataloader) = load_block_csv(
        block_csv, embedding, tokenized_blocks_file
    )

    # Params used by training invocation script from README
    params = ithemal_utils.BaseParameters(
        data=block_csv,
        embed_mode="none",
        embed_file=None,
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    accuracies = []
    losses = []

    correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    for epoch in range(num_epochs):
        training_progress = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"
        )
        for batch in training_progress:
            blocks, throughputs = batch["blocks"], batch["throughput"].to(device)
            optimizer.zero_grad()

            batch_loss = torch.tensor(0.0, device=device)
            for block, throughput in zip(blocks, throughputs):
                # Each datum is a block with a list of instructions
                output = model(block)
                loss = normalized_mse_loss(output, throughput)
                batch_loss += loss
                loss.backward()
                optimizer.step()

                correct += check_correct(output, throughput, tolerance=tolerance).sum()
            
            total += len(blocks)
            accuracy = correct.item() / total.item() if total.item() > 0 else 0
            accuracies.append(accuracy * 100)
            losses.append(batch_loss.item())
            if batch_loss.item():
                training_progress.set_postfix({"Loss": batch_loss.item()})


    print("Training complete.")
    ithemal_utils.dump_model_and_data(model, embedding, predictor_file)
    save_checkpoint(model, optimizer, epoch="final", batch_num=0, filename=model_file)
    print("Model and data saved successfully.")
    plot(accuracies, losses)


def test(
    block_csv,
    predictor_file,
    model_file,
    tokenized_blocks_file,
    tolerance=25,
):
    (model, embedding) = load_model_and_data(predictor_file, model_file)
    (_train_dataloader, test_dataloader) = load_block_csv(
        block_csv, embedding, tokenized_blocks_file
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("Starting evaluation...")
    with torch.no_grad():
        correct = torch.tensor(0.0, device=device)
        total = torch.tensor(0.0, device=device)
        testing_progress = tqdm(test_dataloader, desc="Testing")
        for batch in testing_progress:
            blocks, throughputs = batch["blocks"], batch["throughput"].to(device)
            for block, throughput in zip(blocks, throughputs):
                output = model(block)
                correct += check_correct(output, throughput, tolerance=tolerance).sum()
                total += 1

            accuracy = correct.item() / total.item() if total.item() > 0 else 0
            testing_progress.set_postfix({"Accuracy": f"{accuracy * 100:.2f}%"})


def predict(block_hex, predictor_file, model_file):
    (model, embedding) = load_model_and_data(predictor_file, model_file)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    datum = datum_of_code(embedding, block_hex)
    with torch.no_grad():
        output = model(datum)
        print(f"Predicted throughput for block {block_hex}: {output.item():.4f}")


if __name__ == "__main__":
    block_csv = "hsw.csv"
    predictor_file = "hsw_predictor.dump"
    model_file = "hsw_predictor.mdl"
    tokenized_blocks_file = "hsw_tokenized_blocks.pkl"
    num_epochs = 1
    tolerance = 25

    print(
        f"Config: {block_csv}, {predictor_file}, {model_file}, {tokenized_blocks_file}, num_epochs={num_epochs}, tolerance={tolerance}"
    )

    print("Begin training...")
    train(
        block_csv,
        predictor_file,
        model_file,
        tokenized_blocks_file,
        num_epochs=num_epochs,
        tolerance=tolerance,
    )

    print("Begin testing...")
    test(
        block_csv,
        predictor_file,
        model_file,
        tokenized_blocks_file,
        tolerance=tolerance,
    )

    """
    print("Begin prediction...")
    predict(
        block_hex="4183ff0119c083e00885c98945c4b8010000000f4fc139c2",
        predictor_file="skl_predictor.dump",
        model_file="skl_predictor.mdl",
    )
    """
