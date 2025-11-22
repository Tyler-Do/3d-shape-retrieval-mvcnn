# retrieval.py
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets import CachedMultiViewShapeDataset


def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate_retrieval(feats: torch.Tensor,
                       labels: torch.Tensor,
                       ks=(1, 5, 10)):
    """
    Compute average Precision/Recall/F1 for cosine-similarity retrieval.
    """
    feats = F.normalize(feats, p=2, dim=1)  # L2-normalise embeddings
    N = feats.size(0)
    ks = sorted(ks)

    metrics = {k: {"P": 0.0, "R": 0.0, "F1": 0.0} for k in ks}
    valid_queries = 0

    for i in range(N):
        q_feat = feats[i:i+1]                          # [1, D]
        sims = torch.mm(q_feat, feats.t()).squeeze(0)  # [N]
        sims[i] = -1e9                                 # ignore self

        sorted_idx = sims.argsort(descending=True)

        same_class_total = (labels == labels[i]).sum().item() - 1
        if same_class_total <= 0:
            # No other object of the same class in the DB -> we skip.
            continue

        valid_queries += 1

        for k in ks:
            topk = sorted_idx[:k]
            hits = (labels[topk] == labels[i]).sum().item()

            P = hits / k
            R = hits / same_class_total
            F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

            metrics[k]["P"] += P
            metrics[k]["R"] += R
            metrics[k]["F1"] += F1

    for k in ks:
        for m in ["P", "R", "F1"]:
            metrics[k][m] /= max(1, valid_queries)

    return metrics


def denorm(img_tensor):
    # inverse of Normalize([0.5],[0.5]) -> bring back to [0,1]
    img = img_tensor * 0.5 + 0.5
    return img.cpu().permute(1, 2, 0).numpy()


def retrieve_topk(query_idx, feats, k=5):
    feats = F.normalize(feats, p=2, dim=1)
    q = feats[query_idx:query_idx+1]                 # [1,D]
    sims = torch.mm(q, feats.t()).squeeze(0)         # [N]
    sims[query_idx] = -1e9                           # ignore itself
    vals, idxs = sims.topk(k)
    return idxs, vals


def visualize_retrieval(query_idx, feats, labels, dataset,
                        k=5, view_id=0):
    """
    Show one query object and its top-k nearest neighbours.
    """
    idxs, sims = retrieve_topk(query_idx, feats, k=k)

    fig, axes = plt.subplots(1, k+1, figsize=(3*(k+1), 3))

    # Query image
    q_views, q_label = dataset[query_idx]            # [V,C,H,W]
    q_img = denorm(q_views[view_id])
    axes[0].imshow(q_img)
    axes[0].set_title(f"Query\n{dataset.classes[q_label]}")
    axes[0].axis("off")

    # Nearest neighbours
    for j, (idx, sim) in enumerate(zip(idxs, sims), start=1):
        v, lbl = dataset[idx]
        img = denorm(v[view_id])
        axes[j].imshow(img)

        hit = (lbl == q_label)                       # bool
        prefix = "[✓]" if hit else "[x]"
        axes[j].set_title(f"{prefix} {dataset.classes[lbl]}\nsim={sim:.2f}")
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualize", action="store_true",
                        help="Show qualitative retrieval examples.")
    args = parser.parse_args()

    cfg = load_config("configs/config.yaml")

    # Load embeddings
    data = np.load(cfg["embeddings_path"], allow_pickle=True)
    feats = torch.from_numpy(data["feats"]).float()      # [N,512]
    labels = torch.from_numpy(data["labels"]).long()     # [N]
    classes = list(data["classes"])
    print("Loaded embeddings:", feats.shape)

    # Quantitative retrieval metrics
    ks = cfg.get("retrieval_topk", [1, 5, 10])
    metrics = evaluate_retrieval(feats, labels, ks=ks)
    for k, vals in metrics.items():
        print(f"K={k}: "
              f"Precision={vals['P']:.4f}, "
              f"Recall={vals['R']:.4f}, "
              f"F1={vals['F1']:.4f}")

    # Qualitative visualisation
    if args.visualize:
        cache_root = cfg["cache_root"]
        img_size = int(cfg["img_size"])
        dataset = CachedMultiViewShapeDataset(cache_root, split="test",
                                              image_size=img_size)

        for q in [0, 5, 10, 20, 30]:
            visualize_retrieval(q, feats, labels, dataset,
                                k=5, view_id=0)


if __name__ == "__main__":
    main()
