# SmartMine Ops — Dataset Setup

This directory holds the training and validation images used to train the
EfficientNetB0 classifier for **mineral**, **rock**, and **ore** classification.

> **Note:** Image files are excluded from version control (see `.gitignore`).
> Use `download_datasets.py` to populate this folder.

---

## Dataset Sources

| Class | Dataset | Kaggle Link |
|-------|---------|-------------|
| `mineral` | Mineral Photos | https://www.kaggle.com/datasets/floriangeillon/mineral-photos |
| `rock` | Rock Classification | https://www.kaggle.com/datasets/salmaneunus/rock-classification |
| `ore` | Rock Classification (ore subset) | https://www.kaggle.com/datasets/salmaneunus/rock-classification |

---

## Prerequisites

### 1. Install the Kaggle Python package

```bash
pip install kaggle
```

### 2. Set up Kaggle API credentials

1. Log in to [kaggle.com](https://www.kaggle.com) and go to  
   **Account → API → Create New API Token**.
2. This downloads a `kaggle.json` file containing your credentials.
3. Place it at `~/.kaggle/kaggle.json` (Linux/macOS) or  
   `%USERPROFILE%\.kaggle\kaggle.json` (Windows).
4. Restrict permissions (Linux/macOS only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

Alternatively, export the credentials as environment variables:

```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

---

## Downloading & Organising the Datasets

Run the download script from inside `smartmine/ai-model/data/`:

```bash
cd smartmine/ai-model/data
python download_datasets.py
```

The script will:

1. Download the **mineral-photos** dataset and assign all images to the `mineral` class.
2. Download the **rock-classification** dataset and use a keyword heuristic to
   separate ore images (sub-folders containing words like `ore`, `iron`, `copper`, etc.)
   from rock images.
3. Split images **80 % train / 20 % val** (randomised with a fixed seed for
   reproducibility).
4. Copy organised images into the `train/` and `val/` sub-directories.
5. Remove temporary extraction folders.

> If the ore keyword heuristic does not detect any ore sub-folders, all images
> from the rock-classification dataset are placed into `rock/`. In that case,
> manually move ore images to `data/train/ore/` and `data/val/ore/`.

---

## Resulting Folder Structure

```
smartmine/ai-model/data/
├── README.md                  ← this file
├── download_datasets.py       ← download & organise script
├── train/
│   ├── mineral/               ← mineral images (train split)
│   ├── rock/                  ← rock images (train split)
│   └── ore/                   ← ore images (train split)
└── val/
    ├── mineral/               ← mineral images (val split)
    ├── rock/                  ← rock images (val split)
    └── ore/                   ← ore images (val split)
```

---

## Minimum Recommended Images per Class

| Split | Class | Minimum |
|-------|-------|---------|
| train | mineral | 100 |
| train | rock | 100 |
| train | ore | 100 |
| val | mineral | 25 |
| val | rock | 25 |
| val | ore | 25 |

More images generally improve accuracy. The mineral-photos and
rock-classification datasets together provide several hundred to a few
thousand images per class, which is sufficient for fine-tuning EfficientNetB0.

---

## Next Steps

After populating the `data/` folder, train the model:

```bash
cd smartmine/ai-model
python train_efficientnet.py
```

The trained weights are saved to `models/efficientnet_smartmine.pth` and the
class-name mapping is saved to `models/class_names.json`.
