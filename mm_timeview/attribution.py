import torch
import numpy as np
from typing import Dict, List


def compute_attribution_at_encoder_outputs(
    model,
    x: torch.Tensor,
    Phi: torch.Tensor,
    y_true: torch.Tensor
) -> Dict[str, float]:
    """
    Compute modality attribution at encoder outputs
    
    Args:
        model: MMTimeview model
        x: Input features [1, n_features]
        Phi: Basis matrix [n_time, n_basis]
        y_true: Target trajectory [n_time]

    Returns:
        Dict mapping modality name -> normalized attribution (sums to 1)
    """
    model.train()

    # forward pass with embeddings
    x = x.clone().detach().requires_grad_(False)

    # compute forward pass to get embeddings with gradients
    embeddings = {}
    for mod_name in model._modality_order:
        mod_features = model._extract_modality_features(x, mod_name)
        e = model.encoders[mod_name](mod_features)
        e.retain_grad()
        embeddings[mod_name] = e

    # concatenate
    concat = torch.cat([embeddings[m] for m in model._modality_order], dim=1)
    c = model.fusion(concat)

    # prediction
    y_pred = torch.matmul(Phi, c.squeeze()) + model.bias

    # loss
    loss = torch.mean((y_pred - y_true) ** 2)

    # backward pass
    loss.backward()

    # calculate attribution at encoder outputs
    modality_attrs = {}
    for mod_name, e in embeddings.items():
        if e.grad is not None:
            # Gradient-input product at encoder output
            alpha = e.grad * e.detach()
            # L2 norm
            modality_attrs[mod_name] = torch.norm(alpha, p=2).item()
        else:
            modality_attrs[mod_name] = 0.0

    # Normalize across modalities
    total = sum(modality_attrs.values()) + 1e-10
    normalized = {m: v / total for m, v in modality_attrs.items()}

    return normalized


def compute_modality_attributions(
    model,
    X: torch.Tensor,
    ts: List[np.ndarray],
    ys: List[np.ndarray],
    verbose: bool = False
) -> Dict[str, np.ndarray]:
    """
    Compute modality attributions for all samples.

    Args:
        model: Trained MMTimeview model
        X: Features [n_samples, n_features]
        ts: List of time arrays
        ys: List of target trajectories
        verbose: Print progress

    Returns:
        Dict mapping modality name -> array of attributions [n_samples]
    """
    n_samples = X.shape[0]
    modality_names = model._modality_order

    # initialize output
    all_attrs = {m: np.zeros(n_samples) for m in modality_names}

    device = next(model.parameters()).device

    for i in range(n_samples):
        if verbose and (i + 1) % 100 == 0:
            print(f"Computing attribution {i + 1}/{n_samples}")

        # sample data
        x_i = X[i:i+1].to(device)
        t_i = ts[i]
        y_i = torch.from_numpy(ys[i]).float().to(device)

        # basis matrix
        Phi = torch.from_numpy(model.basis.get_matrix(t_i)).float().to(device)

        # final modality attribution
        model.zero_grad()
        attr = compute_attribution_at_encoder_outputs(model, x_i, Phi, y_i)

        for m in modality_names:
            all_attrs[m][i] = attr[m]

    return all_attrs


def compute_attribution_statistics(
    attributions: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics of attributions across samples
    """
    stats = {}
    for mod_name, values in attributions.items():
        stats[mod_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
    return stats


def print_attribution_summary(stats: Dict[str, Dict[str, float]]):
    print(f"{'Modality':<20} {'Mean':>10} {'Std':>10}")
    for mod_name, s in sorted(stats.items(), key=lambda x: -x[1]['mean']):
        print(f"{mod_name:<20} {s['mean']:>9.1%} {s['std']:>9.1%}")