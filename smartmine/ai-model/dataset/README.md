# Dataset Structure

> **Primary dataset location has moved.**
> Images are now stored in `smartmine/ai-model/data/` and organised into
> **mineral**, **rock**, and **ore** classes.
> See [`smartmine/ai-model/data/README.md`](../data/README.md) for full
> instructions and the `download_datasets.py` script.

---

## Legacy Structure (safe/unsafe — no longer used)

The original `dataset/` folder was used for binary safe/unsafe classification.
It is kept for reference but is **not** used by the current training pipeline.

```
dataset/
├── train/
│   ├── safe/       ← safe mining condition images
│   └── unsafe/     ← unsafe mining condition images
└── val/
    ├── safe/
    └── unsafe/
```
