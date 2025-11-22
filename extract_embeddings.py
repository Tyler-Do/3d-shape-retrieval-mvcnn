# extract_embeddings.py
import os
import yaml
import numpy as np
import torch

from datasets import make_loaders, CachedMultiViewShapeDataset
from models.mvcnn import MVCNN


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config("configs/config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    cache_root = cfg["cache_root"]
    img_size = int(cfg["img_size"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])
    ckpt_path = cfg["checkpoint_path"]
    out_path = cfg["embeddings_path"]

    # We only need the dataset + loader; no val split here.
    test_ds = CachedMultiViewShapeDataset(cache_root, split="test",
                                          image_size=img_size)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    num_classes = len(test_ds.classes)
    model = MVCNN(
        num_classes=num_classes,
        pretrained=True,
        feature_dim=512,
        pooling="max"
    ).to(device)

    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded weights from:", ckpt_path)

    all_feats = []
    all_labels = []

    with torch.no_grad():
        for views, labels in test_loader:
            views = views.to(device)
            _, feats = model(views)      # feats: [B, 512]
            all_feats.append(feats.cpu())
            all_labels.append(labels)

    all_feats = torch.cat(all_feats, dim=0).numpy()      # [N, 512]
    all_labels = torch.cat(all_labels, dim=0).numpy()    # [N]

    np.savez(out_path, feats=all_feats, labels=all_labels,
             classes=np.array(test_ds.classes))
    print(f"Saved embeddings to {out_path}")


if __name__ == "__main__":
    main()
