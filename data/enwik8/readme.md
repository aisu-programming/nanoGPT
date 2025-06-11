# enwik8 dataset

This directory prepares the [enwik8](https://en.wikipedia.org/wiki/Hutter_Prize) corpus for byte-level language modeling.
Running `prepare.py` will download the dataset, split it 90/10 into train and validation sets and save them as `train.bin` and `val.bin`.
The dataset uses raw bytes so the vocabulary size is always 256.

After preprocessing:
- `train.bin` has ~90M tokens
- `val.bin` has ~10M tokens

The dataset was first introduced in "A High-Throughput byte-level language model" (arXiv:1808.04444).
