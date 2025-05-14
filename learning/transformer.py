import itertools
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from main import BasicBlockCSV, check_correct, plot, save_checkpoint
from pytorch.data.data_cost import DataInstructionEmbedding
from pytorch.ithemal import ithemal_utils


class FlattenedBasicBlockCSV(BasicBlockCSV):
    """
    Dataset for basic blocks, but instructions are flattened into a single sequence.
    """

    def __init__(self, csv_file, embedder):
        super().__init__(csv_file, embedder)

    def __getitem__(self, idx):
        throughput = self.df["throughput"].iloc[idx]
        block = self.embedder.data[idx]
        # Concat the different lines of the block into a single sequence
        flat_instrs = list(itertools.chain(*block.x))
        return {"instrs": flat_instrs, "throughput": throughput}

    @staticmethod
    def collate_fn(batch):
        """
        Find the max length for the given batch and right-pad each sequence so they're all the same length.
        Then, create the attention mask for eqch sequence so the model can ignore the padding.
        """
        max_len = max(len(item["instrs"]) for item in batch)
        # Assumes 0 is the padding token, so we've shoehorned that into DataInstructionEmbedding.prepare_data
        padded_instrs = torch.tensor(
            [item["instrs"] + [0] * (max_len - len(item["instrs"])) for item in batch],
        )
        attention_masks = torch.tensor(
            [
                [1] * len(item["instrs"]) + [0] * (max_len - len(item["instrs"]))
                for item in batch
            ]
        )
        throughputs = torch.tensor(
            [item["throughput"] for item in batch], dtype=torch.float32
        )
        return {
            "instrs": padded_instrs,
            "attn_masks": attention_masks,
            "throughput": throughputs,
        }

    def create_loaders(self):
        train_size = int(0.8 * len(self))  # 80% for training
        test_size = len(self) - train_size  # rest for testing
        train_dataset, test_dataset = random_split(self, [train_size, test_size])
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            collate_fn=FlattenedBasicBlockCSV.collate_fn,  # custom collate handles padding and attention masks
            num_workers=16,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=FlattenedBasicBlockCSV.collate_fn,
            num_workers=8,
            pin_memory=True,
        )

        return (train_dataloader, test_dataloader)


class TransformerThroughputPredictor(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        nhead=8,
        num_layers=6,
        dtype=torch.float32,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, dtype=dtype)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dtype=dtype,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regression = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim, dtype=dtype),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, dtype=dtype),
        )

    def forward(self, x, attention_mask=None):
        embedded = self.embedding(x)

        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
            encoded = self.encoder(
                embedded.transpose(0, 1),
                src_key_padding_mask=src_key_padding_mask,
            ).transpose(0, 1)
        else:
            encoded = self.encoder(embedded.transpose(0, 1)).transpose(0, 1)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded)
            sum_embeddings = (encoded * mask_expanded).sum(dim=1)
            count_tokens = attention_mask.sum(dim=1, keepdim=True)
            pooled = sum_embeddings / count_tokens
        else:
            pooled = encoded.mean(dim=1)

        throughput = self.regression(pooled)
        return throughput.squeeze(-1)


def normalized_mse_loss(output, target, eps=1e-8):
    """
    Normalized MSE similar to the one in [main.py](./main.py), but aiming for better
    numerical stability.
    """
    squared_diff = (output - target) ** 2
    normalizer = torch.abs(target) + eps
    normalized_loss = torch.sqrt(squared_diff + eps) / normalizer
    return normalized_loss.mean()


if __name__ == "__main__":
    csv_file = "hsw.csv"
    tokenized_blocks_file = "hsw_tokenized_blocks.pkl"
    predictor_file = "hsw_tsf_predictor.pkl"
    model_file = "hsw_tsf_model.pkl"
    num_epochs = 5
    tolerance = 25

    csv = os.path.join(os.environ["ITHEMAL_HOME"], csv_file)
    if not os.path.exists(csv):
        raise FileNotFoundError(
            f"File {csv} does not exist. Please ensure the path is correct."
        )

    embedder = DataInstructionEmbedding()

    dataset = None
    if os.path.exists(tokenized_blocks_file):
        print(f"Loading dataset from {tokenized_blocks_file}...")
        dataset = FlattenedBasicBlockCSV.load(tokenized_blocks_file, embedder)
    else:
        dataset = FlattenedBasicBlockCSV(csv, embedder)
        dataset.remove_invalid_hex()
        dataset.tokenize_hex()
        dataset.load_into_embedder()
        dataset.process_into_blocks()
        dataset.save(tokenized_blocks_file)
        print(f"Dataset saved to {tokenized_blocks_file}.")
    (train_loader, test_loader) = dataset.create_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TransformerThroughputPredictor(
        vocab_size=len(embedder.token_to_hot_idx),
        embedding_dim=256,
        hidden_dim=256,
        nhead=8,
        num_layers=6,
        dtype=torch.float32,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01
    )

    accuracies = []
    losses = []

    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)
    for epoch in range(num_epochs):
        model.train()
        batch_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in batch_tqdm:
            instrs = batch["instrs"].to(device)
            attn_masks = batch["attn_masks"].to(device)
            throughput = batch["throughput"].to(device)

            optimizer.zero_grad()
            outputs = model(instrs, attn_masks)
            loss = normalized_mse_loss(outputs, throughput, eps=1e-3)
            loss.backward()
            # Try to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            correct += check_correct(outputs, throughput, tolerance).sum()
            total += len(outputs)
            accuracies.append(correct.item() / total.item() * 100)
            losses.append(loss.item())
            batch_tqdm.set_postfix({"Loss": loss.item(), "Acc": accuracies[-1]})

    print("Training complete.")
    plot(accuracies, losses)

    ithemal_utils.dump_model_and_data(model, embedder, predictor_file)
    save_checkpoint(model, optimizer, epoch="final", batch_num=0, filename=model_file)
    print("Model and data saved successfully.")

    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)
    with torch.no_grad():
        model.eval()
        batch_tqdm = tqdm(test_loader, desc="Testing")
        for batch in batch_tqdm:
            instrs = batch["instrs"].to(device)
            attn_masks = batch["attn_masks"].to(device)
            throughput = batch["throughput"].to(device)
            outputs = model(instrs, attn_masks)

            correct += check_correct(outputs, throughput, tolerance).sum()
            total += len(outputs)
            batch_tqdm.set_postfix({"Accuracy": correct.item() / total.item() * 100})

    print(f"Final Accuracy: {correct.item() / total.item() * 100:.2f}%")
