import os
import pandas as pd
import subprocess
import time
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import xml.etree.ElementTree as ET

import common_libs.utilities as ut
from pytorch.data.data_cost import DataInstructionEmbedding, DataItem
from pytorch.ithemal import ithemal_utils

_TOKENIZER = os.path.join(
    os.environ["ITHEMAL_HOME"], "data_collection", "build", "bin", "tokenizer"
)
_fake_intel = "\n" * 500


def datum_of_code(data, block_hex, verbose):
    xml = subprocess.check_output([_TOKENIZER, block_hex, "--token"])
    data.raw_data = [(-1, -1, _fake_intel, xml)]
    data.data = []
    data.prepare_data(fixed=False, progress=False)
    return data.data[-1]


class BasicBlockCSV(Dataset):
    def __init__(self, csv_file, embedder):
        self.blocks = pd.read_csv(
            csv_file,
            header=None,
            sep=r"\s*,\s*",
            engine="python",
            names=["hex", "throughput"],
        )
        self.embedder = embedder

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        hex, throughput = self.blocks.iloc[idx, :]
        datum = datum_of_code(self.embedder, hex, verbose=False)
        return {"datum": datum, "throughput": throughput}

    @staticmethod
    def collate_fn(batch):
        data = [item["datum"] for item in batch]
        throughputs = torch.tensor(
            [item["throughput"] for item in batch], dtype=torch.float32
        )
        return {"datum": data, "throughput": throughputs}


def save_checkpoint(model, optimizer, epoch, batch_num, filename, **rest):
    # type: (int, int, str, **Any) -> None

    state_dict = {
        "epoch": epoch,
        "batch_num": batch_num,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    for k, v in list(rest.items()):
        state_dict[k] = v

    # ensure directory exists
    try:
        os.makedirs(os.path.dirname(filename))
    except OSError:
        pass

    torch.save(state_dict, filename)


def main():
    hsw = os.path.join(os.environ["ITHEMAL_HOME"], "hsw.csv")
    if not os.path.exists(hsw):
        raise FileNotFoundError(
            f"File {hsw} does not exist. Please ensure the path is correct."
        )
    embedding = DataInstructionEmbedding()
    dataset = BasicBlockCSV(hsw, embedding)

    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # rest for testing
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=BasicBlockCSV.collate_fn,
        num_workers=16,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, collate_fn=BasicBlockCSV.collate_fn
    )

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
        embed_file=embed_file,
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

    def normalized_mse_loss(output, target):
        loss = torch.nn.functional.mse_loss(output, target, reduction="none")
        loss = torch.sqrt(loss) / (target + 1e-3)
        return loss.mean()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i, batch in enumerate(train_dataloader):
        embedding, throughputs = batch["datum"], batch["throughput"].to(device)
        optimizer.zero_grad()

        forward_duration = 0.0
        backward_duration = 0.0
        batch_loss = torch.tensor(0.0, device=device)
        for datum, throughput in zip(embedding, throughputs):
            # Each datum is a block with a list of instructions
            forward_start = time.time()
            output = model(datum)
            forward_end = time.time()
            forward_duration += forward_end - forward_start

            loss = normalized_mse_loss(output, throughput)
            batch_loss += loss

            backward_start = time.time()
            loss.backward()
            optimizer.step()
            backward_end = time.time()
            backward_duration += backward_end - backward_start

        avg_batch_loss = batch_loss / len(embedding)
        if i % 10 == 0:
            print(
                f"Epoch {i // 10} (Batch {i}/{len(train_dataloader)}): Loss = {avg_batch_loss.item()}, Forward Time = {forward_duration:.4f}s, Backward Time = {backward_duration:.4f}s"
            )

    print("Training complete.")
    ithemal_utils.dump_model_and_data(model, embedding, "new_trainer_predictor.dump")
    save_checkpoint(
        model, optimizer, epoch="final", batch_num=0, filename="new_trainer_trained.mdl"
    )

    print("Model and data saved successfully.")
    print("Starting evaluation...")
    for batch in test_dataloader:
        embedding, throughputs = batch["datum"], batch["throughput"].to(device)
        with torch.no_grad():
            test_loss = torch.tensor(0.0, device=device)
            for datum, throughput in zip(embedding, throughputs):
                output = model(datum)
                loss = normalized_mse_loss(output, throughput)
                test_loss += loss

        avg_test_loss = test_loss / len(embedding)
        print(f"Test Loss: {avg_test_loss.item()}")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
