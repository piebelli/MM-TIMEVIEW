import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

from timeview.basis import BSplineBasis

class ModalityEncoder(nn.Module):
    """
    Encoder for a single modality (Linear → ReLU → Linear → e^(k))
    """

    def __init__(self, n_features: int, embedding_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim

        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        self._embedding = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode modality features to embedding.

        Args:
            x: Modality features [batch, n_features]

        Returns:
            e: Embedding [batch, embedding_dim]
        """
        self._embedding = self.net(x)
        return self._embedding


class FusionNetwork(nn.Module):

    def __init__(self, total_embedding_dim: int, n_basis: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_basis)
        )

    def forward(self, concat_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Predict B-spline coefficients from concatenated embeddings.

        Args:
            concat_embeddings: [batch, total_embedding_dim]

        Returns:
            c: B-spline coefficients [batch, n_basis]
        """
        return self.net(concat_embeddings)


class MMTimeview(nn.Module):
 
    def __init__(
        self,
        modality_groups: Dict[str, List[int]],
        n_basis: int = 9,
        T: float = 1.0,
        embedding_dim: int = 16,
        hidden_dim: int = 32,
        seed: int = 42
    ):
        """
        Args:
            modality_groups: Dict mapping modality name -> feature indices
            n_basis: Number of B-spline basis functions
            T: Time horizon
            embedding_dim: Embedding dimension per modality
            hidden_dim: Hidden layer dimension
            seed: Random seed
        """
        super().__init__()
        torch.manual_seed(seed)

        self.n_basis = n_basis
        self.T = T
        self.modality_groups = modality_groups

        # reuse B-spline basis from original implementation
        self.basis = BSplineBasis(n_basis, (0, T))

        # one encoder per modality
        self.encoders = nn.ModuleDict()
        total_embedding_dim = 0

        for mod_name, feature_indices in modality_groups.items():
            n_features = len(feature_indices)
            self.encoders[mod_name] = ModalityEncoder(
                n_features=n_features,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim
            )
            total_embedding_dim += embedding_dim

        # fusion network
        self.fusion = FusionNetwork(
            total_embedding_dim=total_embedding_dim,
            n_basis=n_basis,
            hidden_dim=hidden_dim
        )

        # bias term (as in the original implementation)
        self.bias = nn.Parameter(torch.zeros(1))

        # store modality order
        self._modality_order = sorted(modality_groups.keys())

    def _extract_modality_features(self, x: torch.Tensor, mod_name: str) -> torch.Tensor:
        """Extract features for a specific modality."""
        indices = self.modality_groups[mod_name]
        return x[:, indices]

    def forward_with_embeddings(
        self,
        x: torch.Tensor
    ) -> tuple:
        """
        Forward pass that also returns embeddings for attribution.

        Args:
            x: All features [batch, n_features]

        Returns:
            c: B-spline coefficients [batch, n_basis]
            embeddings: Dict of modality embeddings
        """
        # encode each modality
        embeddings = {}
        for mod_name in self._modality_order:
            mod_features = self._extract_modality_features(x, mod_name)
            embeddings[mod_name] = self.encoders[mod_name](mod_features)

        # concatenate embeddings
        concat = torch.cat([embeddings[m] for m in self._modality_order], dim=1)

        # fusion to get B-spline coefficients
        c = self.fusion(concat)

        return c, embeddings

    def forward(self, x: torch.Tensor, Phi: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict trajectory.

        Args:
            x: All features [batch, n_features]
            Phi: Basis matrix [n_time, n_basis]

        Returns:
            y_pred: Predicted trajectory [batch, n_time] or [n_time] if batch=1
        """
        c, _ = self.forward_with_embeddings(x)

        # y(t) = Φ(t) @ c + bias
        if x.shape[0] == 1:
            y_pred = torch.matmul(Phi, c.squeeze()) + self.bias
        else:
            # Batched: [batch, n_time]
            y_pred = torch.matmul(c, Phi.T) + self.bias

        return y_pred

    def predict_trajectory(self, x: torch.Tensor, t: np.ndarray) -> np.ndarray:
        """
        Predict trajectory for given features and time points.

        Args:
            x: Features [n_features] or [1, n_features]
            t: Time points [n_time]

        Returns:
            y: Predicted trajectory [n_time]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        Phi = torch.from_numpy(self.basis.get_matrix(t)).float().to(x.device)

        with torch.no_grad():
            y_pred = self.forward(x, Phi)

        return y_pred.cpu().numpy()


def create_mm_timeview(
    modality_groups: Dict[str, List[int]],
    n_basis: int = 9,
    T: float = 1.0,
    embedding_dim: int = 16,
    seed: int = 42
) -> MMTimeview:
    """
    Convenience function to create MM-TIMEVIEW model.

    Args:
        modality_groups: Dict mapping modality name -> feature indices
        n_basis: Number of B-spline basis functions
        T: Time horizon
        embedding_dim: Embedding dimension per modality
        seed: Random seed

    Returns:
        MMTimeview model
    """
    return MMTimeview(
        modality_groups=modality_groups,
        n_basis=n_basis,
        T=T,
        embedding_dim=embedding_dim,
        seed=seed
    )


def train_mm_timeview(
    model: MMTimeview,
    X: np.ndarray,
    ts: List[np.ndarray],
    ys: List[np.ndarray],
    n_epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 32,
    lambda_smooth: float = 0.1,  # ADD THIS PARAMETER
    verbose: bool = True,
    device: torch.device = None
) -> Dict[str, List[float]]:
    """
    Train MM-TIMEVIEW model

    Args:
        model: MMTimeview model
        X: Features [n_samples, n_features]
        ts: List of time arrays
        ys: List of target trajectories
        n_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        lambda_smooth: Weight for smoothness regularization
        verbose: Print progress
        device: Device to use (default: CPU)

    Returns:
        Training history dict with 'train_loss' key
    """
    if device is None:
        device = torch.device('cpu')

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_samples = len(X)
    history = {'train_loss': []}

    X_tensor = torch.from_numpy(X).float().to(device)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        # shuffle indices
        indices = np.random.permutation(n_samples)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]

            optimizer.zero_grad()
            batch_loss = 0.0

            for idx in batch_indices:
                x_i = X_tensor[idx:idx+1]
                t_i = ts[idx]
                y_i = torch.from_numpy(ys[idx]).float().to(device)

                Phi = torch.from_numpy(model.basis.get_matrix(t_i)).float().to(device)

                # coefficients for smoothness loss
                c, _ = model.forward_with_embeddings(x_i)
                y_pred = torch.matmul(Phi, c.squeeze()) + model.bias

                # MSE loss
                mse_loss = torch.mean((y_pred - y_i) ** 2)
                
                # smoothness loss
                smooth_loss = torch.mean((c[:, 1:] - c[:, :-1]) ** 2)
                
                # combined loss
                batch_loss += mse_loss + lambda_smooth * smooth_loss

            batch_loss = batch_loss / len(batch_indices)
            batch_loss.backward()
            optimizer.step()

            epoch_loss += batch_loss.item() * len(batch_indices)

        avg_loss = epoch_loss / n_samples
        history['train_loss'].append(avg_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1:3d}/{n_epochs} - Loss: {avg_loss:.6f}")

    return history