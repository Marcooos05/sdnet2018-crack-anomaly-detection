from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.manifold import TSNE

import torch
matplotlib.rcParams['figure.dpi'] = 100


def plot_roc_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    auroc: float,
    title: str = 'ROC Curve',
    ax=None,
    label: str = None,
    color: str = 'steelblue',
) -> plt.Axes:
    
    fpr, tpr, _ = roc_curve(labels, scores)
    _label = label or f'AUROC = {auroc:.4f}'

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(fpr, tpr, color=color, lw=2, label=_label)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    return ax



def plot_pr_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    auprc: float,
    title: str = 'Precision-Recall Curve',
    ax=None,
    label: str = None,
    color: str = 'darkorange',
) -> plt.Axes:
    
    precision, recall, _ = precision_recall_curve(labels, scores)
    _label = label or f'AUPRC = {auprc:.4f}'
    baseline = labels.mean()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(recall, precision, color=color, lw=2, label=_label)
    ax.axhline(baseline, color='k', linestyle='--', lw=1, alpha=0.5,
               label=f'Baseline (prevalence = {baseline:.3f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    return ax



def plot_confusion_matrix(
    cm: np.ndarray,
    title: str = 'Confusion Matrix',
    class_names: list = None,
    ax=None,
) -> plt.Axes:
    
    if class_names is None:
        class_names = ['Normal', 'Anomaly']

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=11)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]:,}',
                    ha='center', va='center', fontsize=12,
                    color='white' if cm[i, j] > thresh else 'black')

    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_title(title, fontsize=13)
    return ax


def plot_score_histogram(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float = None,
    title: str = 'Anomaly Score Distribution',
    ax=None,
    bins: int = 60,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    normal_scores  = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    ax.hist(normal_scores,  bins=bins, alpha=0.6, color='steelblue',
            label=f'Normal (n={len(normal_scores):,})',   density=True)
    ax.hist(anomaly_scores, bins=bins, alpha=0.6, color='tomato',
            label=f'Anomaly (n={len(anomaly_scores):,})', density=True)

    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--', lw=1.5,
                   label=f'Threshold τ = {threshold:.4f}')

    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    return ax



def plot_loss_curves(
    train_losses: list,
    val_losses: list = None,
    title: str = 'Training Loss',
    ylabel: str = 'Loss',
    ax=None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, color='steelblue', lw=2, label='Train')
    if val_losses is not None:
        ax.plot(epochs, val_losses, color='darkorange', lw=2, label='Val')

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    return ax



_COLOURS = ['steelblue', 'darkorange', 'green', 'red', 'purple', 'brown']


def plot_roc_multi(
    results: dict,
    title: str = 'ROC Curves — All Models',
    figsize: tuple = (7, 6),
) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(res['labels'], res['scores'])
        ax.plot(fpr, tpr, color=_COLOURS[i % len(_COLOURS)], lw=2,
                label=f"{name}  (AUROC={res['auroc']:.4f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    return fig


def plot_pr_multi(
    results: dict,
    title: str = 'Precision-Recall Curves — All Models',
    figsize: tuple = (7, 6),
) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=figsize)
    for i, (name, res) in enumerate(results.items()):
        prec, rec, _ = precision_recall_curve(res['labels'], res['scores'])
        ax.plot(rec, prec, color=_COLOURS[i % len(_COLOURS)], lw=2,
                label=f"{name}  (AUPRC={res['auprc']:.4f})")
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    return fig



_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def _denorm(tensor: torch.Tensor) -> np.ndarray:
    
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * _IMAGENET_STD + _IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def plot_patches_grid(
    tensors: torch.Tensor,
    labels: torch.Tensor = None,
    pred_labels: torch.Tensor = None,
    scores: np.ndarray = None,
    n_cols: int = 8,
    title: str = '',
    figsize: tuple = None,
) -> plt.Figure:
    
    N      = len(tensors)
    n_rows = (N + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (n_cols * 1.6, n_rows * 1.8)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i in range(N):
        img = _denorm(tensors[i])
        axes[i].imshow(img)
        axes[i].axis('off')

        caption = ''
        if labels is not None:
            caption += f"GT:{int(labels[i])}"
        if pred_labels is not None:
            caption += f" P:{int(pred_labels[i])}"
        if scores is not None:
            caption += f"\n{scores[i]:.3f}"
        if caption:
            color = 'red' if (labels is not None and int(labels[i]) == 1) else 'black'
            axes[i].set_title(caption, fontsize=7, color=color)

    for j in range(N, len(axes)):
        axes[j].axis('off')

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    return fig




def plot_reconstructions(
    inputs: torch.Tensor,
    recons: torch.Tensor,
    labels: torch.Tensor = None,
    scores: np.ndarray = None,
    n: int = 8,
    title: str = 'Reconstructions',
) -> plt.Figure:
    
    n = min(n, len(inputs))
    fig, axes = plt.subplots(2, n, figsize=(n * 1.8, 4))

    for i in range(n):
        axes[0, i].imshow(_denorm(inputs[i]))
        axes[0, i].axis('off')
        axes[1, i].imshow(_denorm(recons[i]))
        axes[1, i].axis('off')

        if scores is not None:
            axes[0, i].set_title(f'GT:{int(labels[i])}' if labels is not None else '',
                                 fontsize=7)
            axes[1, i].set_title(f'err:{scores[i]:.3f}', fontsize=7)

    axes[0, 0].set_ylabel('Input', fontsize=10)
    axes[1, 0].set_ylabel('Recon', fontsize=10)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig



def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str = 't-SNE of Latent Space',
    figsize: tuple = (8, 6),
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
) -> plt.Figure:
    
    print(f"  Running t-SNE on {len(embeddings):,} points (dim={embeddings.shape[1]})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=n_iter,
                random_state=random_state)
    z2d = tsne.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=figsize)
    colours = ['steelblue', 'tomato']
    for cls, name, c in [(0, 'Normal', colours[0]), (1, 'Anomaly', colours[1])]:
        mask = labels == cls
        ax.scatter(z2d[mask, 0], z2d[mask, 1],
                   c=c, s=4, alpha=0.4, label=f'{name} (n={mask.sum():,})',
                   rasterized=True)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.legend(fontsize=10, markerscale=3)
    plt.tight_layout()
    return fig



def plot_preprocessing_grid(
    sample_patches: list,
    lime_patches: list,
    clahe_patches: list,
    labels: list = None,
    title: str = 'Preprocessing Comparison: None vs CLAHE vs LIME',
) -> plt.Figure:
   
    N    = len(sample_patches)
    fig, axes = plt.subplots(3, N, figsize=(N * 2, 7))

    rows  = [sample_patches, clahe_patches, lime_patches]
    names = ['None (Original)', 'CLAHE', 'LIME']

    for row_i, (patches, name) in enumerate(zip(rows, names)):
        for col_i, patch in enumerate(patches):
            axes[row_i, col_i].imshow(patch)
            axes[row_i, col_i].axis('off')
            if col_i == 0:
                axes[row_i, col_i].set_ylabel(name, fontsize=10, rotation=90,
                                               va='center')
            if row_i == 0 and labels is not None:
                lbl = 'Anomaly' if labels[col_i] == 1 else 'Normal'
                axes[row_i, col_i].set_title(lbl, fontsize=9)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig




def plot_bar_ablation(
    names: list,
    values: list,
    ylabel: str = 'AUROC',
    title: str = 'Ablation Results',
    figsize: tuple = (8, 4),
    color: str = 'steelblue',
) -> plt.Figure:
    
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(names, values, color=color, edgecolor='black', alpha=0.8)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Value labels on bars
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=20, ha='right', fontsize=9)
    plt.tight_layout()
    return fig
