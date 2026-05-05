"""
Shared JEPA World Model — Single model for all agents, batch inference.

ALL agents share one encoder + predictor. Individual differences come from
different observations and traits. The PHYSICS is the same for everyone.

Batch processing: one matrix multiply instead of N separate ones. 50x speedup.

Architecture: LeWorldModel (Maes et al. 2026, arXiv:2603.19312)
"""

import numpy as np
from world_model import WorldEncoder, WorldPredictor, SIGReg


class SharedWorldModel:
    """
    Single JEPA world model shared by all agents.

    Batch encode: (N, 40) -> (N, 24) in one matmul.
    Batch CEM plan: vectorized across all agents simultaneously.
    """

    def __init__(self, obs_dim: int = 40, action_dim: int = 8,
                 latent_dim: int = 24):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        hidden_dim = latent_dim * 2
        self.encoder = WorldEncoder(obs_dim, latent_dim, hidden_dim)
        self.predictor = WorldPredictor(latent_dim, action_dim, hidden_dim)
        self.sigreg = SIGReg(n_projections=15)

        self.experience_buffer: list = []
        self.max_buffer_size = 20000
        self.learning_rate = 0.001
        self.lambda_reg = 0.01
        self.train_steps = 0

        # Compatibility: CEM planner for per-agent planning calls
        from world_model import CEMPlanner
        self._compat_planner = CEMPlanner(
            self.predictor, action_dim,
            horizon=2, n_samples=12, n_elites=4, n_iterations=2
        )

    def encode(self, observation: np.ndarray) -> np.ndarray:
        """Single or batch encode. Compatible with Agent.decide_action()."""
        return self.encoder.encode(observation)

    def encode_batch(self, observations: np.ndarray) -> np.ndarray:
        """Batch encode: (N, obs_dim) -> (N, latent_dim). Alias for encode."""
        return self.encoder.encode(observations)

    def predict_next(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Single predict. Compatible with Agent code."""
        return self.predictor.predict(z, action)

    @property
    def planner(self):
        """CEM planner access for Agent.decide_action() compatibility."""
        return self._compat_planner

    def store_experience(self, obs, action, next_obs):
        """Single experience store. Compatible with Agent.execute_action()."""
        self.experience_buffer.append((obs.copy(), action.copy(), next_obs.copy()))
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

    def get_world_understanding(self) -> dict:
        """Compatible with Agent.to_dict()."""
        return self.get_understanding()

    def predict_batch(self, z: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Batch predict: (N, latent_dim) + (N, action_dim) -> (N, latent_dim)."""
        return self.predictor.predict(z, actions)

    def plan_batch(self, z_current: np.ndarray, z_goal: np.ndarray,
                   n_samples: int = 12, n_elites: int = 4,
                   n_iterations: int = 2, horizon: int = 2) -> np.ndarray:
        """
        Vectorized CEM planning for ALL agents simultaneously.

        Instead of N separate CEM loops, ONE batched CEM:
        - Sample (N, n_samples, horizon, action_dim) random actions
        - Rollout through shared predictor in batch
        - Per-agent elite selection and distribution update

        Returns: (N, action_dim) — first action for each agent.
        """
        N = z_current.shape[0]
        if N == 0:
            return np.zeros((0, self.action_dim), dtype=np.float32)

        A = self.action_dim
        S = n_samples
        H = horizon
        K = n_elites

        mu = np.zeros((N, H, A), dtype=np.float32)
        sigma = np.ones((N, H, A), dtype=np.float32) * 0.5

        for _ in range(n_iterations):
            # Sample: (N, S, H, A)
            noise = np.random.randn(N, S, H, A).astype(np.float32)
            actions = mu[:, None, :, :] + sigma[:, None, :, :] * noise
            np.clip(actions, -1.0, 1.0, out=actions)

            # Rollout all N*S trajectories through predictor
            # Start from z_current tiled S times: (N*S, latent_dim)
            z = np.broadcast_to(z_current[:, None, :],
                                (N, S, self.latent_dim)).reshape(N * S, -1).copy()

            for t in range(H):
                a = actions[:, :, t, :].reshape(N * S, A)
                z = self.predictor.predict(z, a)

            # Cost: squared distance to goal in latent space
            z_final = z.reshape(N, S, self.latent_dim)
            costs = np.sum((z_final - z_goal[:, None, :]) ** 2, axis=-1)  # (N, S)

            # Elite selection per agent
            elite_idx = np.argpartition(costs, K, axis=1)[:, :K]  # (N, K)

            # Gather elite actions using advanced indexing
            batch_idx = np.arange(N)[:, None].repeat(K, axis=1)  # (N, K)
            elite_actions = actions[batch_idx, elite_idx]  # (N, K, H, A)

            # Update distribution
            mu = elite_actions.mean(axis=1)
            sigma = elite_actions.std(axis=1) + 0.01

        # Return first action of best sequence
        return mu[:, 0, :]  # (N, A)

    def store_experience_batch(self, obs: np.ndarray, actions: np.ndarray,
                                next_obs: np.ndarray):
        """Store experiences from multiple agents."""
        n = min(len(obs), 50)  # Cap per-tick storage
        for i in range(n):
            self.experience_buffer.append((obs[i].copy(), actions[i].copy(),
                                            next_obs[i].copy()))
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer = self.experience_buffer[-self.max_buffer_size:]

    def train_step(self, batch_size: int = 64):
        """Train shared model using directional finite-difference gradients."""
        if len(self.experience_buffer) < batch_size:
            return

        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        obs = np.array([self.experience_buffer[i][0] for i in indices])
        actions = np.array([self.experience_buffer[i][1] for i in indices])
        next_obs = np.array([self.experience_buffer[i][2] for i in indices])

        z = self.encoder.encode(obs)
        z_next_true = self.encoder.encode(next_obs)
        z_next_pred = self.predictor.predict(z, actions)

        pred_loss = float(np.mean((z_next_pred - z_next_true) ** 2))
        reg_loss = self.sigreg.compute(z) if len(z) > 2 else 0.0
        total_loss = pred_loss + self.lambda_reg * reg_loss

        # Directional gradient estimation (LeWM approach)
        self._estimate_gradients(obs, actions, next_obs)
        self.train_steps += 1

    def _estimate_gradients(self, obs, actions, next_obs):
        """Directional finite-difference gradients (3 directions per weight)."""
        n_directions = 3
        eps = 0.001

        weight_sets = [
            (self.encoder, ['W1', 'W2', 'W3', 'b1', 'b2', 'b3']),
            (self.predictor, ['W_latent', 'W_hidden', 'W_out']),
        ]

        for module, weight_names in weight_sets:
            for wname in weight_names:
                W = getattr(module, wname)
                grad = np.zeros_like(W)

                for _ in range(n_directions):
                    direction = np.random.randn(*W.shape).astype(W.dtype)
                    direction /= (np.linalg.norm(direction) + 1e-10)

                    setattr(module, wname, W + eps * direction)
                    z_p = self.encoder.encode(obs)
                    zn_p = self.predictor.predict(z_p, actions)
                    zt_p = self.encoder.encode(next_obs)
                    loss_plus = float(np.mean((zn_p - zt_p) ** 2))

                    setattr(module, wname, W - eps * direction)
                    z_m = self.encoder.encode(obs)
                    zn_m = self.predictor.predict(z_m, actions)
                    zt_m = self.encoder.encode(next_obs)
                    loss_minus = float(np.mean((zn_m - zt_m) ** 2))

                    grad += ((loss_plus - loss_minus) / (2 * eps)) * direction
                    setattr(module, wname, W)

                grad /= n_directions
                setattr(module, wname, W - self.learning_rate * grad)

    def get_understanding(self) -> dict:
        return {
            "train_steps": self.train_steps,
            "buffer_size": len(self.experience_buffer),
            "latent_dim": self.latent_dim,
            "model_maturity": min(1.0, self.train_steps / 500.0),
        }
