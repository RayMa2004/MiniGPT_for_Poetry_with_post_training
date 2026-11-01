# a sample should be a full poem

from pathlib import Path
from functools import partial

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import CharTokenizer


def collate_batch(batch, pad_id):
    """将一个 Batch 内的样本补齐到当前 Batch 的最大长度。"""

    batch_size = len(batch)
    max_len = max(x.size(0) for x, y in batch)

    x_batch = torch.full(
        (batch_size, max_len),
        pad_id,
        dtype=torch.long,
    )

    y_batch = torch.full(
        (batch_size, max_len),
        -100,
        dtype=torch.long,
    )

    for i, (x, y) in enumerate(batch):
        length = x.size(0)
        x_batch[i, :length] = x
        y_batch[i, :length] = y

    return x_batch, y_batch


class PoetryDataset(Dataset): 
    # a data management system
    def __init__(self, file_path, tokenizer, block_size=192):
        self.tokenizer = tokenizer
        self.block_size = block_size # the upper bound of input

        with open(file_path, "r", encoding="utf-8") as f:
            poems = [line.strip() for line in f if line.strip()]
            # poems = [full_poem 1, full_poem 2, ...]
        
        self.samples = []
        skipped = 0 # count the invalid data

        for poem in poems: # encode each poem
            ids = tokenizer.encode(poem)

            if len(ids) < 2:
                continue # x = ids[:-1] needs at least len = 2
            if len(ids) > block_size + 1: # NOTICE: ids <= block_size + 1
                skipped += 1
                continue

            self.samples.append(ids)

        print(
            f"{file_path}: {len(self.samples)} samples,"
            f"skipped {skipped} long poems"
        )

    def __len__(self): # amount of samples
        return len(self.samples)

    def __getitem__(self, idx):
        ids = self.samples[idx]

        # 暂时不补齐，交给 collate_batch 统一处理
        x = ids[:-1]
        y = ids[1:]

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )
