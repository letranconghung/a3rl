import torch
import torch.nn as nn
import torch.nn.functional as F


class DensityNetwork(nn.Module):
    """
    Density ratio estimation network for offline-to-online RL.

    Estimates the density ratio between offline and online data distributions
    using a two-sample density ratio estimation approach.

    Architecture: MLP with (256, 256) hidden dims, ReLU activation, no layer norm,
    and Softplus output activation for positive outputs.
    """

    def __init__(self, obs_dim, action_dim, hidden_dims=(256, 256), lr=3e-4, device='cuda'):
        super().__init__()
        self.device = device

        # Build MLP: (obs+action) -> hidden -> hidden -> 1 -> Softplus
        input_dim = obs_dim + action_dim
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Final layer outputs scalar
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())

        self.network = nn.Sequential(*layers)
        self.to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through density network.

        Args:
            obs: Observations tensor of shape (batch_size, obs_dim)
            action: Actions tensor of shape (batch_size, action_dim)

        Returns:
            Density weights of shape (batch_size, 1), positive values via Softplus
        """
        x = torch.cat([obs, action], dim=-1)
        return self.network(x)

    def update(self, offline_batch: dict, online_batch: dict) -> dict:
        """
        Update density network using two-sample density ratio estimation.

        Loss follows sac_learner_priority.py:348-379:
            offline_f_star = -log(2.0 / (offline_weight + 1))
            online_f_prime = log(2 * online_weight / (online_weight + 1))
            loss = mean(offline_f_star - online_f_prime)

        Args:
            offline_batch: Batch from offline replay buffer with 'obs', 'actions'
            online_batch: Batch from online replay buffer with 'obs', 'actions'

        Returns:
            Dictionary with metrics: density_loss, offline_weight, online_weight
        """
        # Compute density weights for both batches
        offline_weight = self.forward(offline_batch['obs'], offline_batch['actions'])
        online_weight = self.forward(online_batch['obs'], online_batch['actions'])

        # Compute f-divergence terms (from density ratio estimation)
        # Add epsilon for numerical stability to avoid log(0)
        eps = 1e-10
        offline_f_star = -torch.log(2.0 / (offline_weight + 1) + eps)
        online_f_prime = torch.log(2 * online_weight / (online_weight + 1) + eps)

        # Loss: minimize difference (trains to distinguish offline vs online)
        loss = torch.mean(offline_f_star - online_f_prime)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'density_loss': loss.item(),
            'offline_weight': offline_weight.mean().item(),
            'online_weight': online_weight.mean().item(),
            'offline_weight_tensor': offline_weight.detach(),  # For priority updates
        }
