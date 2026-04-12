from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              f1_score, confusion_matrix)



def train_svdd_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device)
        z = model(x)
        loss = ((z - model.centre) ** 2).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)


def train_ae_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
) -> float:
    
    model.train()
    criterion = nn.MSELoss()
    total_loss = 0.0
    for x, _ in loader:
        x = x.to(device)
        recon, _ = model(x)
        loss = criterion(recon, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    if scheduler is not None:
        scheduler.step()

    return total_loss / len(loader)


def eval_ae_epoch(
    model,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate autoencoder reconstruction loss (no gradient)."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            recon, _ = model(x)
            total_loss += criterion(recon, x).item()
    return total_loss / len(loader)



def train_classifier_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device,
    scheduler=None,
) -> tuple[float, float]:
    
    model.train()
    total_loss   = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.float().to(device)

        logits = model(x)
        loss   = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss    += loss.item()
        preds          = (torch.sigmoid(logits) >= 0.5).long()
        total_correct += (preds == y.long()).sum().item()
        total_samples += y.size(0)

    if scheduler is not None:
        scheduler.step()

    acc = total_correct / total_samples
    return total_loss / len(loader), acc


def eval_classifier_epoch(
    model,
    loader: DataLoader,
    criterion,
    device: torch.device,
) -> tuple[float, float]:
   
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.float().to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            total_loss    += loss.item()
            preds          = (torch.sigmoid(logits) >= 0.5).long()
            total_correct += (preds == y.long()).sum().item()
            total_samples += y.size(0)

    return total_loss / len(loader), total_correct / total_samples




@torch.no_grad()
def eval_scores(
    model,
    loader: DataLoader,
    device: torch.device,
    score_fn: str = 'anomaly_score',
) -> tuple[np.ndarray, np.ndarray]:
    
    model.eval()
    fn = getattr(model, score_fn)
    all_scores, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        s = fn(x).cpu().numpy()
        all_scores.append(s)
        all_labels.append(y.numpy())
    return np.concatenate(all_scores), np.concatenate(all_labels)




def calibrate_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    n_thresholds: int = 200,
) -> tuple[float, float]:
    
    thresholds = np.linspace(scores.min(), scores.max(), n_thresholds)
    best_tau, best_f1 = 0.0, 0.0
    for tau in thresholds:
        preds = (scores >= tau).astype(int)
        f1    = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1  = f1
            best_tau = tau
    return float(best_tau), float(best_f1)



def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict:
    
    preds  = (scores >= threshold).astype(int)
    auroc  = roc_auc_score(labels, scores)
    auprc  = average_precision_score(labels, scores)
    f1     = f1_score(labels, preds, zero_division=0)
    cm     = confusion_matrix(labels, preds)
    return {
        'auroc': auroc,
        'auprc': auprc,
        'f1':    f1,
        'cm':    cm,
        'preds': preds,
    }


def print_metrics(metrics: dict, model_name: str = '') -> None:
    
    tag = f"[{model_name}] " if model_name else ""
    print(f"{tag}AUROC: {metrics['auroc']:.4f}  "
          f"AUPRC: {metrics['auprc']:.4f}  "
          f"F1: {metrics['f1']:.4f}")
    cm = metrics['cm']
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0,0]:6d}  FP={cm[0,1]:6d}")
    print(f"    FN={cm[1,0]:6d}  TP={cm[1,1]:6d}")



@torch.no_grad()
def build_patchcore_memory(
    extractor,
    loader: DataLoader,
    device: torch.device,
    coreset_ratio: float = 0.10,
    patches_per_image: int = 32,
    max_images: int | None = None,
    max_embeddings: int | None = 50_000,
    projection_dim: int = 128,
    seed: int = 42,
) -> np.ndarray:
    
    extractor.eval()
    rng = np.random.default_rng(seed)

    # Collect a manageable subset of local descriptors from each normal image.
    all_feats = []
    total_images = 0
    for x, _ in loader:
        if max_images is not None and total_images >= max_images:
            break
        x = x.to(device)
        tokens = extractor.extract_patch_tokens(x)       # (B, P, D)
        tokens = F.normalize(tokens, p=2, dim=2).cpu().numpy()
        if max_images is not None and total_images + tokens.shape[0] > max_images:
            tokens = tokens[:max_images - total_images]
        total_images += tokens.shape[0]

        for sample_tokens in tokens:
            if patches_per_image is not None and patches_per_image < len(sample_tokens):
                idx = rng.choice(len(sample_tokens), size=patches_per_image, replace=False)
                sample_tokens = sample_tokens[idx]
            all_feats.append(sample_tokens)

    all_feats = np.concatenate(all_feats, axis=0)   # (N_tokens, D)

    if max_embeddings is not None and len(all_feats) > max_embeddings:
        keep_idx = rng.choice(len(all_feats), size=max_embeddings, replace=False)
        all_feats = all_feats[keep_idx]

    n_total   = len(all_feats)
    n_coreset = max(1, int(n_total * coreset_ratio))
    print(f"  Images processed: {total_images:,}")
    print(f"  Patch descriptors: {n_total:,} → coreset: {n_coreset:,} "
          f"({coreset_ratio*100:.0f}%)")

    # Greedy farthest-first traversal coreset subsampling on a projected space.
    coreset_idx = _greedy_coreset(
        all_feats,
        n_coreset,
        projection_dim=projection_dim,
        seed=seed,
    )
    return all_feats[coreset_idx]


def _greedy_coreset(
    features: np.ndarray,
    n: int,
    projection_dim: int | None = 128,
    seed: int = 42,
) -> np.ndarray:
    
    if n >= len(features):
        return np.arange(len(features))

    work_feats = features
    if projection_dim is not None and features.shape[1] > projection_dim:
        rng = np.random.default_rng(seed)
        proj = rng.standard_normal((features.shape[1], projection_dim)).astype(np.float32)
        proj /= np.sqrt(projection_dim)
        work_feats = features @ proj

    N = len(features)
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(N))]
    min_dists = np.full(N, np.inf)

    for _ in range(n - 1):
        # Update distances to nearest coreset point
        last   = work_feats[selected[-1]]              # (D,)
        dists  = np.linalg.norm(work_feats - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        # Pick farthest point
        next_idx = int(np.argmax(min_dists))
        selected.append(next_idx)
        min_dists[next_idx] = 0.0   # mark as selected

    return np.array(selected)


def patchcore_scores(
    extractor,
    loader: DataLoader,
    memory_bank: np.ndarray,
    device: torch.device,
    image_score: str = 'max',
    top_k_patches: int = 5,
    patch_batch_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    
    extractor.eval()
    mem_t = torch.as_tensor(memory_bank, dtype=torch.float32, device=device)
    all_scores, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            tokens = extractor.extract_patch_tokens(x)    # (B, P, D)
            tokens = F.normalize(tokens, p=2, dim=2)
            bsz, n_patches, feat_dim = tokens.shape
            flat = tokens.reshape(bsz * n_patches, feat_dim)

            nn_parts = []
            for start in range(0, flat.shape[0], patch_batch_size):
                chunk = flat[start:start + patch_batch_size]
                dists = torch.cdist(chunk, mem_t)
                nn_parts.append(dists.min(dim=1).values)

            patch_scores = torch.cat(nn_parts).reshape(bsz, n_patches)
            if image_score == 'mean_topk':
                k = min(top_k_patches, n_patches)
                score = torch.topk(patch_scores, k=k, dim=1).values.mean(dim=1)
            else:
                score = patch_scores.max(dim=1).values

            all_scores.append(score.cpu().numpy())
            all_labels.append(y.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)



def save_checkpoint(model, path: str, extra: dict = None) -> None:
    payload = {'state_dict': model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"  Saved checkpoint → {path}")


def load_checkpoint(model, path: str) -> dict:
    payload = torch.load(path, map_location='cpu')
    model.load_state_dict(payload['state_dict'])
    print(f"  Loaded checkpoint ← {path}")
    return payload
