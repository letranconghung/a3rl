import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, activation = nn.ReLU, use_layer_norm = True, activate_final = False, hidden_dims = (256, 256)):
        super().__init__()
        input_dim = obs_dim + action_dim
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dims[-1], 1))
        if activate_final:
            layers.append(activation())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        input = torch.cat([obs, action], dim = -1)
        return self.network(input)

class EnsembleCritic(nn.Module):
    def __init__(self, num_qs = 2, num_min_qs = 2, **critic_kwargs):
        super().__init__()
        self.num_qs = num_qs
        self.num_min_qs = num_min_qs
        self.ensemble = nn.ModuleList([Critic(**critic_kwargs) for _ in range(num_qs)])
        
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_values = torch.stack([critic(obs, action) for critic in self.ensemble], dim = 0)

        return q_values
        
    def sample_min_qs(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_values = self.forward(obs, action)
        indices = torch.randperm(self.num_qs, device = q_values.device)[:self.num_min_qs]
        sampled_qs = q_values[indices]
        min_q = sampled_qs.min(dim = 0)[0]
        return min_q

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, activation = nn.ReLU, hidden_dims = (256, 256),
                 log_std_min = -20, log_std_max = 2):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], action_dim)
        
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
            Returns mean and log_std of the action distribution
        """
        h = self.trunk(obs)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        """

        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        if deterministic:
            action = torch.tanh(mean)
            log_prob = torch.zeros((obs.shape[0], 1), device = obs.device)
        else:
            normal = torch.distributions.Normal(mean, std)
            x = normal.rsample()
            action = torch.tanh(x)

            log_prob = normal.log_prob(x)
            log_prob -= torch.log(torch.clamp(1 - action.pow(2), min = 1e-6))
            log_prob = log_prob.sum(dim = -1, keepdim = True)
        
        return action, log_prob

class SAC(nn.Module):
    def __init__(self, obs_dim, action_dim, device = 'cuda', hidden_dims = (256, 256), num_qs = 2, num_min_qs = 2, actor_lr = 3e-4, critic_lr = 3e-4, alpha_lr = 3e-4, discount = 0.99, tau = 0.005, use_layer_norm = True):
        super().__init__()
        self.device = device
        self.discount = discount
        self.tau = tau
        self.num_qs = num_qs
        self.num_min_qs = num_min_qs
        self.target_entropy = - 0.5 * action_dim
    
        self.actor = Actor(obs_dim, action_dim, hidden_dims = hidden_dims).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), actor_lr)
        
        critic_kwargs = {
            'obs_dim': obs_dim,
            'action_dim': action_dim,
            'hidden_dims': hidden_dims,
            'use_layer_norm': use_layer_norm
        }
        
        self.critic = EnsembleCritic(num_qs = num_qs, num_min_qs=num_min_qs, **critic_kwargs).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), critic_lr)

        self.target_critic = EnsembleCritic(num_qs = num_qs, num_min_qs = num_min_qs, **critic_kwargs).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        for param in self.target_critic.parameters():
            param.requires_grad = False

        # Learnable temperature
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def update_critic(self, batch, weights=None):
        """
        Update both critic and target critic.

        Args:
            batch: Dictionary with obs, next_obs, actions, rewards, dones
            weights: Optional tensor of shape (batch_size,) for weighted loss.
                     If None, uses uniform weighting (standard MSE).
                     Weights should sum to 1 for proper scaling.
        """
        obs = batch['obs']
        next_obs = batch['next_obs']
        actions = batch['actions']
        rewards = batch['rewards'].unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)
        dones = batch['dones'].unsqueeze(-1)  # (batch_size,) -> (batch_size, 1)
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_obs)
            target_q = self.target_critic.sample_min_qs(next_obs, next_actions)

            target_q = target_q - self.alpha * next_log_probs

            target_q = rewards + (1 - dones) * self.discount * target_q

        q_values = self.critic(obs, actions)

        # Compute weighted MSE loss
        if weights is None:
            # Standard MSE (uniform weighting)
            critic_loss = F.mse_loss(q_values, target_q.unsqueeze(0).expand(self.num_qs, -1, -1))
        else:
            # Weighted MSE: weights should be (batch_size,), reshape to (1, batch_size, 1)
            # Weights sum to 1.0, so weighted sum = weighted average (same scale as mean)
            weights = weights.view(1, -1, 1)
            td_errors = (q_values - target_q.unsqueeze(0).expand(self.num_qs, -1, -1)) ** 2
            # Weighted average over batch, mean over ensemble and output dim
            critic_loss = (td_errors * weights).sum(dim=1).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()


        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'q_mean': q_values.mean().item(),
            'q_values': q_values
        }

    def update_actor(self, batch, weights=None):
        """
        Update actor network.

        Args:
            batch: Dictionary with obs
            weights: Optional tensor of shape (batch_size,) for weighted loss.
                     If None, uses uniform weighting.
                     Weights should sum to 1 for proper scaling.
        """
        obs = batch['obs']
        actions, log_probs = self.actor.sample(obs)
        q_values = self.critic(obs, actions)
        q_mean = q_values.mean(dim=0)

        # Per-sample actor loss
        per_sample_loss = self.alpha.detach() * log_probs - q_mean

        if weights is None:
            # Standard mean (uniform weighting)
            actor_loss = per_sample_loss.mean()
            entropy = -log_probs.mean().item()
        else:
            # Weighted loss
            weights = weights.view(-1, 1)
            actor_loss = (per_sample_loss * weights).sum()
            entropy = -(log_probs * weights).sum().item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {
            'actor_loss': actor_loss.item(),
            'entropy': entropy,
            'q_values_policy': q_values
        }

    def update_temperature(self, entropy):
        """Update temperature parameter"""
        # Loss: -log_alpha * (entropy - target_entropy).detach()
        temp_loss = -self.log_alpha * (entropy - self.target_entropy)

        self.alpha_optimizer.zero_grad()
        temp_loss.backward()
        self.alpha_optimizer.step()

        return {
            'alpha': self.alpha.item(),
            'temp_loss': temp_loss.item()
        }

    def update(self, batch):
        critic_info = self.update_critic(batch)
        actor_info = self.update_actor(batch)
        temp_info = self.update_temperature(actor_info['entropy'])
        return {**critic_info, **actor_info, **temp_info}