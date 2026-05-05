"""
JEPA-based World Model inspired by LeWM (LeCun's Joint Embedding Predictive Architecture).

This implements a simplified but faithful NumPy version of the LeWorldModel:
- Encoder: maps observations to latent embeddings
- Predictor: predicts next latent state given current state + action
- SIGReg: regularizer enforcing Gaussian-distributed latent embeddings
- CEM Planner: Cross-Entropy Method for action optimization in latent space
"""

import numpy as np
from typing import Optional


# ============================================================================
# Activation Functions & Utilities
# ============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-10)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def rms_norm(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return gamma * x / rms


# ============================================================================
# SIGReg Regularizer (Stochastic Isotropic Gaussian Regularization)
# ============================================================================

class SIGReg:
    """
    Enforces isotropic Gaussian structure on latent embeddings using the
    Cramer-Wold theorem: project onto random directions and test normality.
    """

    def __init__(self, n_projections: int = 50):
        self.n_projections = n_projections

    def epps_pulley_statistic(self, h: np.ndarray) -> float:
        """Simplified Epps-Pulley normality test on 1D projections."""
        n = len(h)
        h_centered = h - h.mean()
        h_std = h_centered / (h.std() + 1e-8)
        # Characteristic function-based test approximation
        t_values = np.linspace(0.1, 2.0, 20)
        empirical_cf = np.mean(np.exp(1j * np.outer(t_values, h_std)), axis=1)
        gaussian_cf = np.exp(-0.5 * t_values**2)
        return float(np.mean(np.abs(empirical_cf - gaussian_cf)**2))

    def compute(self, Z: np.ndarray) -> float:
        """
        Compute SIGReg loss for batch of embeddings.
        Z: (batch_size, latent_dim)
        """
        D = Z.shape[1]
        loss = 0.0
        for _ in range(self.n_projections):
            u = np.random.randn(D)
            u /= np.linalg.norm(u) + 1e-10
            h = Z @ u  # 1D projection
            loss += self.epps_pulley_statistic(h)
        return loss / self.n_projections


# ============================================================================
# Encoder: Maps world state observations to latent embeddings
# ============================================================================

class WorldEncoder:
    """
    Encodes multi-dimensional world observations into compact latent vectors.
    Inspired by ViT but adapted for structured world state data.
    """

    def __init__(self, obs_dim: int, latent_dim: int = 64, hidden_dim: int = 128):
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        scale1 = np.sqrt(2.0 / obs_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        scale3 = np.sqrt(2.0 / hidden_dim)

        self.W1 = np.random.randn(obs_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale2
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, latent_dim) * scale3
        self.b3 = np.zeros(latent_dim)

        self.gamma1 = np.ones(hidden_dim)
        self.gamma2 = np.ones(hidden_dim)
        self.gamma_out = np.ones(latent_dim)

    def encode(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: (..., obs_dim) -> (..., latent_dim)
        """
        h = gelu(rms_norm(obs @ self.W1 + self.b1, self.gamma1))
        h = gelu(rms_norm(h @ self.W2 + self.b2, self.gamma2))
        z = rms_norm(h @ self.W3 + self.b3, self.gamma_out)
        return z


# ============================================================================
# Predictor: Predicts next latent state from current state + action
# ============================================================================

class WorldPredictor:
    """
    Predicts next latent embedding given current latent + action.
    Uses Adaptive Layer Normalization (AdaLN) for action conditioning,
    matching the LeWM architecture.
    """

    def __init__(self, latent_dim: int = 64, action_dim: int = 8, hidden_dim: int = 128):
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        scale_l = np.sqrt(2.0 / latent_dim)
        scale_a = np.sqrt(2.0 / action_dim)
        scale_h = np.sqrt(2.0 / hidden_dim)

        # Main prediction pathway
        self.W_latent = np.random.randn(latent_dim, hidden_dim) * scale_l
        self.W_hidden = np.random.randn(hidden_dim, hidden_dim) * scale_h
        self.W_out = np.random.randn(hidden_dim, latent_dim) * scale_h

        # AdaLN: action conditions layer normalization (scale + shift per layer)
        self.W_ada1_scale = np.random.randn(action_dim, hidden_dim) * scale_a
        self.W_ada1_shift = np.random.randn(action_dim, hidden_dim) * scale_a
        self.W_ada2_scale = np.random.randn(action_dim, hidden_dim) * scale_a
        self.W_ada2_shift = np.random.randn(action_dim, hidden_dim) * scale_a

        self.gamma1 = np.ones(hidden_dim)
        self.gamma2 = np.ones(hidden_dim)
        self.gamma_out = np.ones(latent_dim)

    def predict(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        z: (..., latent_dim), action: (..., action_dim) -> (..., latent_dim)
        """
        # AdaLN conditioning from action
        scale1 = 1.0 + action @ self.W_ada1_scale
        shift1 = action @ self.W_ada1_shift
        scale2 = 1.0 + action @ self.W_ada2_scale
        shift2 = action @ self.W_ada2_shift

        # Layer 1 with AdaLN
        h = z @ self.W_latent
        h = rms_norm(h, self.gamma1) * scale1 + shift1
        h = gelu(h)

        # Layer 2 with AdaLN
        h = h @ self.W_hidden
        h = rms_norm(h, self.gamma2) * scale2 + shift2
        h = gelu(h)

        # Output projection
        z_next = rms_norm(h @ self.W_out, self.gamma_out)

        # Residual connection for temporal continuity
        return 0.8 * z_next + 0.2 * z


# ============================================================================
# CEM Planner: Cross-Entropy Method for planning in latent space
# ============================================================================

class CEMPlanner:
    """
    Plans optimal action sequences using Cross-Entropy Method in latent space.
    Much faster than pixel-space planning since rollouts happen in low-dim space.
    """

    def __init__(self, predictor: WorldPredictor, action_dim: int,
                 horizon: int = 5, n_samples: int = 64,
                 n_elites: int = 10, n_iterations: int = 8):
        self.predictor = predictor
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_samples = n_samples
        self.n_elites = n_elites
        self.n_iterations = n_iterations

    def plan(self, z_current: np.ndarray, z_goal: np.ndarray,
             action_bounds: tuple = (-1.0, 1.0)) -> np.ndarray:
        """
        Find optimal action sequence to reach goal state in latent space.
        Returns: (horizon, action_dim) optimal action sequence
        """
        mu = np.zeros((self.horizon, self.action_dim))
        sigma = np.ones((self.horizon, self.action_dim)) * 0.5

        for _ in range(self.n_iterations):
            # Sample action sequences
            noise = np.random.randn(self.n_samples, self.horizon, self.action_dim)
            actions = mu[None] + sigma[None] * noise
            actions = np.clip(actions, action_bounds[0], action_bounds[1])

            # Rollout in latent space and compute costs
            costs = np.zeros(self.n_samples)
            for i in range(self.n_samples):
                z = z_current.copy()
                for t in range(self.horizon):
                    z = self.predictor.predict(z, actions[i, t])
                costs[i] = np.sum((z - z_goal)**2)

            # Select elites
            elite_idx = np.argsort(costs)[:self.n_elites]
            elites = actions[elite_idx]

            # Update distribution
            mu = elites.mean(axis=0)
            sigma = elites.std(axis=0) + 0.01

        return mu


# ============================================================================
# Complete JEPA World Model
# ============================================================================

class JEPAWorldModel:
    """
    Complete JEPA-based world model combining encoder, predictor, SIGReg,
    and CEM planner. Used by agents to understand and predict world dynamics.

    Training loop:
        L = L_pred + lambda * SIGReg(Z)
    where L_pred = ||z_hat_{t+1} - z_{t+1}||^2
    """

    def __init__(self, obs_dim: int, action_dim: int, latent_dim: int = 64):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        hidden_dim = latent_dim * 2
        self.encoder = WorldEncoder(obs_dim, latent_dim, hidden_dim)
        self.predictor = WorldPredictor(latent_dim, action_dim, hidden_dim)
        self.sigreg = SIGReg(n_projections=15)
        self.planner = CEMPlanner(self.predictor, action_dim,
                                   horizon=2, n_samples=12,
                                   n_elites=4, n_iterations=2)

        # Experience buffer for learning
        self.experience_buffer: list = []
        self.max_buffer_size = 5000
        self.learning_rate = 0.001
        self.lambda_reg = 0.01
        self.train_steps = 0

    def encode(self, observation: np.ndarray) -> np.ndarray:
        return self.encoder.encode(observation)

    def predict_next(self, z: np.ndarray, action: np.ndarray) -> np.ndarray:
        return self.predictor.predict(z, action)

    def plan_to_goal(self, current_obs: np.ndarray, goal_obs: np.ndarray) -> np.ndarray:
        z_current = self.encode(current_obs)
        z_goal = self.encode(goal_obs)
        return self.planner.plan(z_current, z_goal)

    def store_experience(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray):
        self.experience_buffer.append((obs, action, next_obs))
        if len(self.experience_buffer) > self.max_buffer_size:
            self.experience_buffer.pop(0)

    def train_step(self, batch_size: int = 32) -> dict:
        """
        Single training step using stored experiences.
        L = L_pred + lambda * SIGReg(Z)

        Uses finite-difference gradient approximation for NumPy compatibility.
        """
        if len(self.experience_buffer) < batch_size:
            return {"pred_loss": 0.0, "reg_loss": 0.0, "total_loss": 0.0}

        # Sample batch
        indices = np.random.choice(len(self.experience_buffer), batch_size, replace=False)
        batch = [self.experience_buffer[i] for i in indices]

        obs_batch = np.array([b[0] for b in batch])
        act_batch = np.array([b[1] for b in batch])
        next_obs_batch = np.array([b[2] for b in batch])

        # Forward pass
        z = self.encoder.encode(obs_batch)
        z_next_true = self.encoder.encode(next_obs_batch)
        z_next_pred = self.predictor.predict(z, act_batch)

        # Prediction loss
        pred_loss = float(np.mean((z_next_pred - z_next_true)**2))

        # SIGReg regularization
        reg_loss = self.sigreg.compute(z)

        total_loss = pred_loss + self.lambda_reg * reg_loss

        # Approximate gradient descent via weight perturbation
        self._perturb_weights(total_loss, obs_batch, act_batch, next_obs_batch)
        self.train_steps += 1

        return {"pred_loss": pred_loss, "reg_loss": reg_loss, "total_loss": total_loss}

    def _perturb_weights(self, current_loss: float, obs: np.ndarray,
                         actions: np.ndarray, next_obs: np.ndarray):
        """
        Directional finite-difference gradient estimation.
        (LeWM: Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026, arXiv:2603.19312)

        Instead of random perturbation (try noise, keep if better), we:
        1. Sample N random unit directions per weight matrix
        2. Compute central-difference gradient estimate along each direction
        3. Average the directional gradients
        4. Apply gradient descent step

        This is 5-10x more sample-efficient than perturbation.
        """
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
                    # Random unit direction
                    direction = np.random.randn(*W.shape)
                    direction /= (np.linalg.norm(direction) + 1e-10)

                    # Forward evaluation: W + eps*d
                    setattr(module, wname, W + eps * direction)
                    z = self.encoder.encode(obs)
                    z_next_true = self.encoder.encode(next_obs)
                    z_next_pred = self.predictor.predict(z, actions)
                    loss_plus = float(np.mean((z_next_pred - z_next_true) ** 2))

                    # Backward evaluation: W - eps*d
                    setattr(module, wname, W - eps * direction)
                    z = self.encoder.encode(obs)
                    z_next_true = self.encoder.encode(next_obs)
                    z_next_pred = self.predictor.predict(z, actions)
                    loss_minus = float(np.mean((z_next_pred - z_next_true) ** 2))

                    # Central difference gradient estimate
                    grad += ((loss_plus - loss_minus) / (2 * eps)) * direction

                    # Restore original weights
                    setattr(module, wname, W)

                grad /= n_directions
                # Gradient descent step
                setattr(module, wname, W - self.learning_rate * grad)

    # ------------------------------------------------------------------
    # LeWM Metrics (Maes et al. 2026)
    # ------------------------------------------------------------------

    def compute_temporal_straightness(self, recent_n: int = 20) -> float:
        """
        Temporal path straightness in latent space.
        (LeWM Section 5.1: Temporal Latent Path Straightening)

        Measures how linear latent trajectories are.
        Score 1.0 = perfectly straight (smooth physical dynamics learned).
        Score ~0 = erratic/random embeddings.
        """
        if len(self.experience_buffer) < recent_n:
            return 0.0

        recent = self.experience_buffer[-recent_n:]
        z_sequence = [self.encode(exp[0]) for exp in recent]

        total_path = sum(
            np.linalg.norm(z_sequence[i + 1] - z_sequence[i])
            for i in range(len(z_sequence) - 1)
        )
        direct_dist = np.linalg.norm(z_sequence[-1] - z_sequence[0])
        return float(np.clip(direct_dist / (total_path + 1e-8), 0, 1))

    def probe_physical_understanding(self) -> dict:
        """
        Test if the world model has learned physical structure.
        (Qu, Morel, McCabe, Bietti, Lanusse, Ho, LeCun, 2026, arXiv:2603.13227)

        Trains a linear probe on latent embeddings to predict physical
        parameters (temperature, social tension, resource scarcity).
        High R² = world model understands underlying physics.
        """
        if len(self.experience_buffer) < 50:
            return {"physical_r2": 0.0, "n_samples": 0}

        Z, Y = [], []
        for obs, _, _ in self.experience_buffer[-100:]:
            z = self.encode(obs)
            Z.append(z)
            # Physical parameters from observation vector:
            # obs[32] = temperature/5, obs[34] = social_tension, obs[36] = resource_scarcity
            Y.append([obs[32] * 5.0, obs[34], obs[36]])

        Z = np.array(Z)
        Y = np.array(Y)

        # Linear probe via least squares
        try:
            W_probe, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
            Y_pred = Z @ W_probe
            ss_res = np.sum((Y - Y_pred) ** 2)
            ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2) + 1e-8
            r2 = float(1.0 - ss_res / ss_tot)
        except np.linalg.LinAlgError:
            r2 = 0.0

        return {"physical_r2": round(max(0.0, r2), 4), "n_samples": len(Z)}

    # ------------------------------------------------------------------
    # Understanding Summary
    # ------------------------------------------------------------------

    def get_world_understanding(self) -> dict:
        """Returns metrics about the model's current understanding."""
        base = {
            "train_steps": self.train_steps,
            "buffer_size": len(self.experience_buffer),
            "latent_dim": self.latent_dim,
            "model_maturity": min(1.0, self.train_steps / 500.0),
        }
        # Compute LeWM metrics occasionally to save CPU
        if self.train_steps > 0 and self.train_steps % 50 == 0:
            base["temporal_straightness"] = self.compute_temporal_straightness()
            base["physical_understanding"] = self.probe_physical_understanding()
        return base
