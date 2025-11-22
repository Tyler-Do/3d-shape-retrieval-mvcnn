# 3D Shape Retrieval using View-based MVCNN

Course: M2 Machine Vision and AI, Université Paris-Saclay  
Author: Manh Tuan Do

This repository contains the code for a view-based 3D shape retrieval system on ModelNet10.

### Repository structure

This repository contains a minimal but complete pipeline for view-based 3D shape retrieval on ModelNet10.

```text
3d-shape-retrieval-mvcnn/
├── configs/
│   └── modelnet10.yaml           # Paths, number of views, training hyperparameters
│
├── data/
│   ├── ModelNet10/               # Raw CAD meshes (downloaded separately)
│   └── cache/                    # Rendered & cached PNG views (created by render_cache.py)
│
├── models/
│   └── mvcnn.py                  # MVCNN (ResNet-18 backbone + view pooling)
│
├── datasets.py                   # PyTorch Dataset / DataLoader for cached multi-view images
├── render_cache.py               # Render meshes with trimesh/OpenGL and save PNG views
├── train_mvcnn.py                # Training & validation loop, best-checkpoint saving
├── extract_embeddings.py         # Freeze MVCNN and export 512-d embeddings for all shapes
├── retrieval.py                  # Retrieval metrics (Precision@K, Recall@K, F1) & visualisation
│
├── figures/                      # Example plots (learning curves, qualitative results)
│   ├── learning_curves.png
│   ├── classification_samples.png
│   └── retrieval_examples.png
│
├── requirements.txt              # Python dependencies (PyTorch, torchvision, trimesh, …)
└── README.md                     # This file
```

We deliberately keep the structure simple: one script per stage of the pipeline (render → cache → train → extract → retrieve). This makes it easy to re-run or modify individual components without touching the others.

---

### How to reproduce the experiments

Below we describe how a new user can reproduce our results step by step.
All commands are meant to be run from the root of the repository.

#### 1. Create environment and install dependencies

```bash
# (optional but recommended)
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Main dependencies:

* Python ≥ 3.8
* PyTorch + torchvision (GPU build if available)
* `trimesh`, `PyOpenGL`, `numpy`, `matplotlib`, `tqdm`, `pyyaml`

#### 2. Download ModelNet10

1. Download the official **ModelNet10** dataset (offered by Princeton ModelNet).
2. Unzip it so that you obtain a folder `ModelNet10/` containing subfolders
   `bed`, `chair`, `desk`, …
3. Update the dataset path in `configs/modelnet10.yaml`, for example:

```yaml
data_root: /absolute/path/to/data/ModelNet10
cache_root: data/cache/modelnet10
num_views: 8
img_size: 128
```

We keep all non-rendered meshes under `data/ModelNet10/` and all rendered PNGs under `data/cache/`.

#### 3. Render and cache multi-view images

```bash
python render_cache.py --config configs/modelnet10.yaml
```

This script:

* iterates over all meshes in ModelNet10,
* renders `V` views per object (by default `V = 8`) at resolution `128×128`,
* saves them as `PNG` files under:

```text
data/cache/modelnet10/{train|test}/{class_name}/{object_id}/view_XX.png
```

This is the **offline indexing** stage: we pay the cost of rendering only once.

> On Google Colab: this step can be slow; we monitor RAM and, if needed, render
> only the training split first and the test split later.

#### 4. Train the MVCNN classifier

```bash
python train_mvcnn.py --config configs/modelnet10.yaml
```

The script:

* builds a `CachedMultiViewShapeDataset` over the cached PNGs,
* instantiates an MVCNN with a ResNet-18 backbone and view-pooling,
* trains with cross-entropy + Adam, using the hyperparameters from the config
  (learning rate, weight decay, batch size, number of epochs),
* logs training/validation loss and accuracy,
* saves the best model checkpoint to the path specified in the config,
  e.g. `checkpoints/mvcnn_modelnet10_best.pth`.

The resulting learning-curve figure (see `figures/learning_curves.png`) helps us check for overfitting and convergence.

#### 5. Extract 512-d embeddings

```bash
python extract_embeddings.py --config configs/modelnet10.yaml --split test
```

This script:

* loads the best checkpoint,
* freezes the MVCNN and removes the classification head,
* forwards all views of each object and applies view pooling,
* stores a 512-dimensional embedding and the corresponding label for every shape
  into a NumPy / PyTorch file, e.g.:

```text
embeddings/test_feats.pt
embeddings/test_labels.pt
```

We use exactly these embeddings for all retrieval experiments.

#### 6. Run retrieval and visualisation

To compute retrieval metrics and generate qualitative figures:

```bash
python retrieval.py --config configs/modelnet10.yaml --split test --visualize
```

This script:

* loads the stored embeddings, L2-normalises them,
* for each query shape, computes cosine similarity with all database shapes,
* evaluates Precision@K, Recall@K and F1@K (for K = 1, 5, 10 by default),
* saves retrieval grids such as `figures/retrieval_examples.png`, where the
  left column is the query and the top-K neighbours are shown on the right,
  annotated with similarity scores.

These figures are the ones we reproduce in the report (classification examples and retrieval examples).

#### 7. Colab-specific notes

When running on **Google Colab**:

* Make sure to select **Runtime → Change runtime type → GPU**.
* Mount Google Drive and point `data_root`, `cache_root` and `checkpoint_path`
  in the config to folders under `/content/drive/MyDrive/…` to persist results.
* If RAM is limited, we can reduce `batch_size` in the config or render fewer
  views per object (`num_views`).

