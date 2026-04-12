from __future__ import annotations

import torch


@torch.no_grad()
def k_center_greedy(
    features: torch.Tensor,
    sampling_ratio: float = 0.1,
    progress: bool = False,
) -> tuple[torch.Tensor, int]:
    if features.ndim != 2:
        raise ValueError(f"features must have shape (N, D), got {tuple(features.shape)}")

    n_total = features.shape[0]
    n_select = max(1, int(n_total * sampling_ratio))
    if n_select >= n_total:
        return features, n_total

    work = features.detach()
    device = work.device
    selected = torch.empty(n_select, dtype=torch.long, device=device)
    selected[0] = torch.randint(n_total, (1,), device=device)

    min_dists = torch.full((n_total,), torch.inf, device=device)
    for i in range(1, n_select):
        last = work[selected[i - 1]].unsqueeze(0)
        dists = torch.cdist(work, last, p=2.0).squeeze(1)
        min_dists = torch.minimum(min_dists, dists)
        min_dists[selected[:i]] = 0.0
        selected[i] = torch.argmax(min_dists)

    return work[selected], n_select
