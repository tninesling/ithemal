import concurrent
from concurrent.futures import ProcessPoolExecutor
import itertools
import multiprocessing
import os
import pandas as pd
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from pytorch.data.data_cost import DataInstructionEmbedding

_TOKENIZER = os.path.join(
    os.environ["ITHEMAL_HOME"], "data_collection", "build", "bin", "tokenizer"
)
_fake_intel = "\n" * 500

def tokenize_single(hex_str):
    return subprocess.check_output([_TOKENIZER, hex_str, "--token"])

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

        self.df = self.df.iloc[:100000]

        self.embedder = embedder

    ''' Dataset related methods '''
    def __len__(self):
        return len(self.embedder.data)

    def __getitem__(self, idx):
        throughput = self.df['throughput'].iloc[idx]
        block = self.embedder.data[idx]
        # Concat the different lines of the block into a single sequence
        flat_instrs = list(itertools.chain(*block.x))
        return {"instrs": flat_instrs, "throughput": throughput}

    @staticmethod
    def collate_fn(batch):
        '''
        Find the max length for the given batch and right-pad each sequence so they're all the same length.
        Then, create the attention mask for eqch sequence so the model can ignore the padding.
        '''
        max_len = max(len(item["instrs"]) for item in batch)
        padded_instrs = torch.tensor(
            [item["instrs"] + [0] * (max_len - len(item["instrs"])) for item in batch],
        )
        attention_masks = torch.tensor(
            [[1] * len(item["instrs"]) + [0] * (max_len - len(item["instrs"])) for item in batch]
        )
        throughputs = torch.tensor(
            [item["throughput"] for item in batch], dtype=torch.float32
        )
        return {"instrs": padded_instrs, "attn_masks": attention_masks, "throughput": throughputs}

    ''' Builder methods '''
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
            futures = {executor.submit(tokenize_single, hex_val): i 
                    for i, hex_val in enumerate(hex_values)}
            
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
            collate_fn=BasicBlockCSV.collate_fn,  # custom collate handles padding and attention masks
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


class TransformerThroughputPredictor(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=hidden_dim
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.regression = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, attention_mask=None):
        # x: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len] (1 for real tokens, 0 for padding)
        
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Convert attention mask for transformer
        if attention_mask is not None:
            # Create a mask for padded positions (1 = position to mask)
            src_key_padding_mask = (attention_mask == 0)
            encoded = self.encoder(
                embedded.transpose(0, 1),  # [seq_len, batch_size, embedding_dim]
                src_key_padding_mask=src_key_padding_mask
            ).transpose(0, 1)  # [batch_size, seq_len, embedding_dim]
        else:
            encoded = self.encoder(
                embedded.transpose(0, 1)
            ).transpose(0, 1)
        
        # Only average over real tokens using the mask
        if attention_mask is not None:
            # Expand mask to match embedding dim
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(encoded)
            # Sum embeddings where mask is 1, and divide by number of real tokens
            sum_embeddings = (encoded * mask_expanded).sum(dim=1)  # [batch_size, embedding_dim]
            count_tokens = attention_mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
            pooled = sum_embeddings / count_tokens  # [batch_size, embedding_dim]
        else:
            pooled = encoded.mean(dim=1)  # [batch_size, embedding_dim]
        
        throughput = self.regression(pooled)  # [batch_size, 1]
        return throughput.squeeze(-1)  # [batch_size]

if __name__ == "__main__":
    csv = os.path.join(os.environ["ITHEMAL_HOME"], "hsw.csv")
    if not os.path.exists(csv):
        raise FileNotFoundError(
            f"File {blocks} does not exist. Please ensure the path is correct."
        )

    embedder = DataInstructionEmbedding()
    blocks = BasicBlockCSV(csv, embedder)
    (train_loader, test_loader) = blocks.remove_invalid_hex().tokenize_hex().load_into_embedder().process_into_blocks().create_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = TransformerThroughputPredictor(vocab_size=len(embedder.token_to_hot_idx), embedding_dim=256, hidden_dim=256).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    batch_tqdm = tqdm(train_loader, desc="Training")
    for batch in batch_tqdm:
        instrs = batch["instrs"].to(device)
        attn_masks = batch["attn_masks"].to(device)
        throughput = batch["throughput"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(instrs, attn_masks)
        loss = nn.MSELoss()(outputs, throughput)

        # Backward pass and optimization
        model.zero_grad()
        loss.backward()
        optimizer.step()

        batch_tqdm.set_postfix({"Loss": loss.item()})

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

            is_correct = torch.abs((outputs - throughput) * 100 / (throughput + 1e-3)) < 25 # correct if within 25%
            correct += is_correct.sum()
            total += len(outputs)
            batch_tqdm.set_postfix({"Accuracy": correct.item() / total.item() * 100})

    print(f"Final Accuracy: {correct.item() / total.item() * 100:.2f}%")
