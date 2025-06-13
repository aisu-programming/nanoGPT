import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from itertools import cycle
from model import T64
import argparse
import shutil



# ==== 訓練參數 ====
BATCH_SIZE = 64
BLOCK_SIZE = 256    # 512
EMBED_DIM = 256    # 512
N_LAYER = 12    # 64
N_HEAD = 2
N_TARGETS = 2
MAX_STEPS = 4_000_000
EVAL_INTERVAL = 10_000
EVAL_BATCH_SIZE = 32
LR = 1e-2
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
DROPOUT_PROB = 0.55
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



# ==== 讀取資料 ====
data_dir = "data/enwik8"
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint8, mode="r")
val_data   = np.memmap(os.path.join(data_dir, "val.bin"  ), dtype=np.uint8, mode="r")
test_data  = np.memmap(os.path.join(data_dir, "test.bin" ), dtype=np.uint8, mode="r")

class CharDatasetMultiTarget(Dataset):
    def __init__(self, data, block_size, n_targets):
        self.data = data
        self.block_size = block_size
        self.n_targets = n_targets

    def __len__(self):
        return len(self.data) - self.block_size - self.n_targets + 1

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx : idx + self.block_size], dtype=torch.long)
        y = torch.stack([
            torch.tensor(self.data[idx + k : idx + k + self.block_size], dtype=torch.long)
            for k in range(1, self.n_targets + 1)
        ], dim=-1)    # shape: (T, K)
        return x, y

train_loader = DataLoader(
    CharDatasetMultiTarget(train_data, BLOCK_SIZE, n_targets=2),
    batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)



# ==== 建立模型 ====
model = T64(
    vocab_size=256,
    block_size=BLOCK_SIZE,
    embd_dim=EMBED_DIM,
    n_layer=N_LAYER,
    n_head=N_HEAD,
    n_targets=N_TARGETS,
    max_steps=MAX_STEPS,
    dropout=DROPOUT_PROB,
).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)



# ==== 顯示模型參數資訊 ====
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_size_MB = trainable_params * 4 / (1024**2)  # assume float32 = 4 bytes

print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Estimated model size: {model_size_MB:.2f} MB")



# ==== 備份程式碼 ====
parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument("--log_dirname", type=str, required=True)
args = parser.parse_args()
shutil.copy("model.py", args.log_dirname)
shutil.copy("train_t64.py", args.log_dirname)



# ==== 評估函數 ====
def evaluate(data, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, len(data) - BLOCK_SIZE - 1, BLOCK_SIZE * EVAL_BATCH_SIZE):
            x_batch = []
            y_batch = []
            for b in range(EVAL_BATCH_SIZE):
                start = i + b * BLOCK_SIZE
                end = start + BLOCK_SIZE
                if end >= len(data): break
                x = torch.tensor(data[start   : end  ], dtype=torch.long)
                y = torch.tensor(data[start+1 : end+1], dtype=torch.long)
                x_batch.append(x)
                y_batch.append(y)
            if not x_batch: break
            x = torch.stack(x_batch).to(device)    # (B, T)
            y = torch.stack(y_batch).to(device)    # (B, T)
            loss = model(x, targets=y).item()
            total_loss += loss * x.numel()
            total_tokens += x.numel()
    avg_loss = total_loss / total_tokens
    bpc = avg_loss / math.log(2)
    return avg_loss, bpc



class AvgMeter:
    def __init__(self, length):
        self.reset()
        self.length = length

    def reset(self):
        self.values = []

    def update(self, value):
        if len(self.values) < self.length:
            self.values.append(value)
        else:
            self.values.pop(0)
            self.values.append(value)

    @property
    def avg(self):
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)



# ==== 訓練主程式 ====
model.train()
train_loader = cycle(train_loader)
pbar = trange(MAX_STEPS)
# pbar = range(MAX_STEPS)
train_loss_meter, train_bpc_meter = AvgMeter(100), AvgMeter(100)
for step in pbar:
    x, y = next(train_loader)
    x, y = x.to(DEVICE), y.to(DEVICE)
    optimizer.zero_grad()
    loss = model(x, targets=y, step=step)
    loss.backward()
    loss = loss.item()
    bpc = loss / math.log(2)
    train_loss_meter.update(loss)
    train_bpc_meter.update(bpc)
    optimizer.step()

    # pbar.set_description(f"Step {step+1:,}: train loss = {train_loss_meter.avg:.4f}, train bpc = {train_bpc_meter.avg:.4f}")
    if (step+1) % 1000 == 0:
        print(f"Step {step+1:4,}: train loss = {train_loss_meter.avg:.4f}, train bpc = {train_bpc_meter.avg:.4f}", flush=True)

    if (step+1) % EVAL_INTERVAL == 0:
        val_loss, val_bpc = evaluate(val_data, DEVICE)
        test_loss, test_bpc = evaluate(test_data, DEVICE)
        print('')
        print(f"Step {step+1:4,}: val   loss = {val_loss:.4f}, val   bpc = {val_bpc:.4f}")
        print(f"Step {step+1:4,}: test  loss = {test_loss:.4f}, test  bpc = {test_bpc:.4f}")
        print('')
        model.train()
