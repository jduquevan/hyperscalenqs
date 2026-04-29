from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Optional

import hydra
import wandb
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from pathlib import Path
import orbax.checkpoint as ocp

os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")

import jax
import optax
import jax.numpy as jnp
import netket as nk
from flax import linen as nn
from flax.training import train_state
from functools import partial


@dataclass
class Args:
    seed: int = 42
    jax_platform_name: str = "gpu"

    # J1-J2 square lattice
    L: int = 10          # lattice side length; N = L*L
    pbc: bool = True
    J1: float = 1.0
    J2: float = 0.5

    # Sampling / AR setup
    n_samples: int = 2048
    machine_pow: int = 2

    # Optimizer
    lr: float = 1e-5
    final_lr: float = 1e-7
    n_iter: int = 1_000_000
    optimizer: str = "adam"
    sgd_momentum: float = 0.0
    decay_rate: float = 0.5
    transition_steps: int = 100000
    use_phase_jacobian_baseline: bool = True
    phase_jacobian_baseline_eps: float = 1e-8

    # PPO / clipped objectives
    ppo_epochs: int = 4
    normalize_advantage: bool = True
    ppo_clip_eps: float = 1e-3

    # Phase channel
    phase_loss_type: str = "delta_clip"   # {"delta_clip", "ratio"}
    phase_coef: float = 1.0
    phase_scale: float = math.pi
    phase_delta_clip: float = 0.3
    phase_ratio_tau: float = 1e-3
    phase_clip_eps: float = 1e-2
    center_imag_advantage: bool = True
    normalize_imag_advantage: bool = True
    phase_delta_l2_coef: float = 0.0

    # Architecture
    model_type: str = "gru2d"   # {"gru1d", "gru2d"}
    embed_dim: int = 32
    rnn_hidden: int = 256
    head_hidden: int = 256
    n_gru_layers: int = 3
    phase_init_std: float = 1e-3

    # Logging / eval
    eval_every: int = 200
    eval_n_samples: int = 1048576
    eval_batch_size: int = 2048
    eval_n_discard_per_chain: int = 0

    wandb_project: str = "hyperscalenqs-j1j2-square"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_mode: str = "online"
    log_every: int = 10
    log_gradient_info: bool = False
    save_checkpoint_every: int = 0   # 0 = only save final; >0 = also save every N iters
    output_directory: Optional[str] = "."
    wandb_directory: Optional[str] = "."


ConfigStore.instance().store(name="config", node=Args)


class TrainState(train_state.TrainState):
    pass


def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_scale(tree, scalar):
    return jax.tree_util.tree_map(lambda x: scalar * x, tree)


def tree_l2_sq(tree):
    return sum(jnp.sum(x * x) for x in jax.tree_util.tree_leaves(tree))


def tree_cast_like(tree, like_tree):
    return jax.tree_util.tree_map(lambda x, y: x.astype(y.dtype), tree, like_tree)


def make_tx(cfg: Args):
    lr_schedule = optax.schedules.cosine_onecycle_schedule(
        transition_steps=cfg.transition_steps,
        peak_value=1e-4,
        pct_start=0.15,
        div_factor=10.0,
        final_div_factor=200.0,
    )

    opt = cfg.optimizer.lower()
    if opt == "sgd":
        tx = optax.sgd(learning_rate=lr_schedule, momentum=cfg.sgd_momentum)
    elif opt == "adam":
        tx = optax.adam(learning_rate=lr_schedule)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    return tx, lr_schedule


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class GRULayer(nn.Module):
    """1-D GRU scan over a sequence (B, T, input_dim) -> (B, T, hidden_size)."""
    hidden_size: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        B, T, _ = x.shape
        carry0 = jnp.zeros((B, self.hidden_size), dtype=x.dtype)

        ScanGRU = nn.scan(
            nn.GRUCell,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=1,
            out_axes=1,
        )
        gru = ScanGRU(features=self.hidden_size)
        _, h = gru(carry0, x)
        return h


def _amp_phase_towers(
    x: jax.Array,
    head_hidden: int,
    local_size: int,
    machine_pow: int,
    phase_scale: float,
    phase_init_std: float,
) -> jax.Array:
    """Shared amplitude + phase tower; returns log_psi (complex)."""
    # amplitude
    x_amp = nn.Dense(head_hidden, name="amp_fc1")(x)
    x_amp = nn.gelu(x_amp)
    x_amp = nn.Dense(head_hidden, name="amp_fc2")(x_amp)
    x_amp = nn.gelu(x_amp)
    amp_logits = nn.Dense(local_size, name="amp_head")(x_amp)
    amp_logp = jax.nn.log_softmax(amp_logits, axis=-1)
    amp_logpsi = amp_logp / float(machine_pow)

    # phase
    x_phase = nn.Dense(head_hidden, name="phase_fc1")(x)
    x_phase = nn.gelu(x_phase)
    x_phase = nn.Dense(head_hidden, name="phase_fc2")(x_phase)
    x_phase = nn.gelu(x_phase)
    phase_raw = nn.Dense(
        local_size,
        name="phase_head",
        kernel_init=nn.initializers.normal(stddev=phase_init_std),
        bias_init=nn.initializers.zeros_init(),
    )(x_phase)
    phase_raw = phase_raw - jnp.mean(phase_raw, axis=-1, keepdims=True)
    phase = phase_scale * jnp.tanh(phase_raw)

    return amp_logpsi + 1j * phase


# ---------------------------------------------------------------------------
# Model 1: AR-GRU with 2-D row/column positional embeddings
# ---------------------------------------------------------------------------

class RecurrentAR2DEmbedding(nk.models.AbstractARNN):
    """
    1-D GRU autoregressive model with factorised 2-D positional embeddings.

    Sites are indexed in raster (row-major) order: site i -> row i//L, col i%L.
    The positional embedding is  row_emb(i//L) + col_emb(i%L), giving the
    model spatial awareness at the input level with fewer parameters than a
    flat N-way embedding.
    """
    hilbert: object
    L: int = 10
    embed_dim: int = 32
    rnn_hidden: int = 256
    head_hidden: int = 256
    n_gru_layers: int = 3
    machine_pow: int = 2
    phase_scale: float = math.pi
    phase_init_std: float = 1e-3

    @nn.compact
    def conditionals_log_psi(self, inputs: jax.Array) -> jax.Array:
        if inputs.ndim == 1:
            inputs = inputs[None, :]

        B, N = inputs.shape
        L = self.L

        spin_tokens = (inputs > 0).astype(jnp.int32)
        bos_token = jnp.full((B, 1), 2, dtype=jnp.int32)
        prev_tokens = jnp.concatenate([bos_token, spin_tokens[:, :-1]], axis=1)

        tok_emb = nn.Embed(num_embeddings=3, features=self.embed_dim, name="tok_emb")
        x = tok_emb(prev_tokens)  # (B, N, embed_dim)

        # 2-D positional embedding: row_emb + col_emb
        site_ids = jnp.arange(N, dtype=jnp.int32)
        row_ids = site_ids // L   # (N,)
        col_ids = site_ids % L    # (N,)
        row_emb = nn.Embed(num_embeddings=L, features=self.embed_dim, name="row_emb")
        col_emb = nn.Embed(num_embeddings=L, features=self.embed_dim, name="col_emb")
        pos = row_emb(row_ids) + col_emb(col_ids)   # (N, embed_dim)
        x = x + pos[None, :, :]                      # broadcast over batch

        x = nn.Dense(self.rnn_hidden, name="input_proj")(x)
        x = nn.tanh(x)
        x = nn.LayerNorm(name="input_ln")(x)

        for layer_idx in range(self.n_gru_layers):
            h = GRULayer(hidden_size=self.rnn_hidden, name=f"gru_{layer_idx}")(x)
            x = x + h

        return _amp_phase_towers(
            x,
            head_hidden=self.head_hidden,
            local_size=self.hilbert.local_size,
            machine_pow=self.machine_pow,
            phase_scale=self.phase_scale,
            phase_init_std=self.phase_init_std,
        )


# ---------------------------------------------------------------------------
# Model 2: 2-D GRU recurrence
# ---------------------------------------------------------------------------

class GRUCell2D(nn.Module):
    """
    A single GRU cell whose input is the concatenation of:
      - the token embedding  x_i          (embed_dim after projection)
      - the left hidden state h_left       (rnn_hidden)
      - the top  hidden state h_top        (rnn_hidden)

    The cell state is a single rnn_hidden vector (h_new), which acts as
    both the output and the new left/top carry.
    """
    hidden_size: int

    @nn.compact
    def __call__(
        self,
        h_left: jax.Array,   # (B, hidden_size)
        h_top:  jax.Array,   # (B, hidden_size)
        x:      jax.Array,   # (B, input_dim)
    ) -> jax.Array:           # (B, hidden_size)
        inp = jnp.concatenate([x, h_left, h_top], axis=-1)  # (B, input_dim + 2*hidden)
        # Project to hidden_size so that GRUCell sees the right input dim
        inp_proj = nn.Dense(self.hidden_size, name="inp_proj")(inp)
        cell = nn.GRUCell(features=self.hidden_size, name="gru_cell")
        # GRUCell carry is the hidden state itself
        _, h_new = cell(h_left, inp_proj)
        return h_new


class GRULayer2D(nn.Module):
    """
    A single 2-D GRU layer using two nested lax.scans.

    Input  x: (B, L, L, d)  – site embeddings in raster order
    Output h: (B, L, L, hidden_size)

    Outer scan: iterates over rows (sequential).
      carry = top-row hidden states, shape (B, L, hidden_size)
    Inner scan: iterates over columns (sequential) within each row.
      carry = h_left, shape (B, hidden_size)

    At position (r, c):
      h_top  = top_carry[:, c, :]   (hidden state at (r-1, c))
      h_left = column carry          (hidden state at (r,   c-1))
    """
    hidden_size: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        B, L, _, d = x.shape

        cell = GRUCell2D(hidden_size=self.hidden_size, name="cell")

        def row_fn(top_carry, x_row):
            # top_carry: (B, L, hidden_size) – hidden states of previous row
            # x_row:     (B, L, d)           – embeddings for current row

            def col_fn(h_left, inputs):
                x_c, h_top_c = inputs   # (B, d), (B, hidden_size)
                h_new = cell(h_left, h_top_c, x_c)
                return h_new, h_new

            h_left0 = jnp.zeros((B, self.hidden_size), dtype=x_row.dtype)
            # Transpose x_row and top_carry to (L, B, ...) for scan over columns
            x_row_T = jnp.transpose(x_row, (1, 0, 2))        # (L, B, d)
            top_T   = jnp.transpose(top_carry, (1, 0, 2))    # (L, B, hidden_size)

            _, h_row_T = jax.lax.scan(col_fn, h_left0, (x_row_T, top_T))
            h_row = jnp.transpose(h_row_T, (1, 0, 2))        # (B, L, hidden_size)
            return h_row, h_row

        top0 = jnp.zeros((B, L, self.hidden_size), dtype=x.dtype)
        # Transpose x to (L, B, L, d) for scan over rows
        x_rows = jnp.transpose(x, (1, 0, 2, 3))              # (L, B, L, d) – rows first

        _, h_grid = jax.lax.scan(row_fn, top0, x_rows)       # (L, B, L, hidden_size)
        h_grid = jnp.transpose(h_grid, (1, 0, 2, 3))         # (B, L, L, hidden_size)
        return h_grid


class RecurrentAR2DGRU(nk.models.AbstractARNN):
    """
    Autoregressive model with a true 2-D GRU recurrence.

    Each hidden state at site (r, c) is computed from:
      - the token embedding at (r, c)
      - the hidden state at (r, c-1)  [left]
      - the hidden state at (r-1, c)  [top]

    This gives the model direct access to the immediately-above hidden state,
    which the 1-D GRU can only reach after L intermediate steps.

    Multiple 2-D GRU layers are stacked with residual connections, matching
    the design of RecurrentAR2DEmbedding.
    """
    hilbert: object
    L: int = 10
    embed_dim: int = 32
    rnn_hidden: int = 256
    head_hidden: int = 256
    n_gru_layers: int = 3
    machine_pow: int = 2
    phase_scale: float = math.pi
    phase_init_std: float = 1e-3

    @nn.compact
    def conditionals_log_psi(self, inputs: jax.Array) -> jax.Array:
        if inputs.ndim == 1:
            inputs = inputs[None, :]

        B, N = inputs.shape
        L = self.L

        spin_tokens = (inputs > 0).astype(jnp.int32)
        bos_token = jnp.full((B, 1), 2, dtype=jnp.int32)
        prev_tokens = jnp.concatenate([bos_token, spin_tokens[:, :-1]], axis=1)

        tok_emb = nn.Embed(num_embeddings=3, features=self.embed_dim, name="tok_emb")
        x = tok_emb(prev_tokens)  # (B, N, embed_dim)

        # Same 2-D positional embedding as RecurrentAR2DEmbedding
        site_ids = jnp.arange(N, dtype=jnp.int32)
        row_ids = site_ids // L
        col_ids = site_ids % L
        row_emb = nn.Embed(num_embeddings=L, features=self.embed_dim, name="row_emb")
        col_emb = nn.Embed(num_embeddings=L, features=self.embed_dim, name="col_emb")
        pos = row_emb(row_ids) + col_emb(col_ids)
        x = x + pos[None, :, :]

        x = nn.Dense(self.rnn_hidden, name="input_proj")(x)
        x = nn.tanh(x)
        x = nn.LayerNorm(name="input_ln")(x)

        # Reshape to (B, L, L, rnn_hidden) for 2-D recurrence
        x_2d = x.reshape(B, L, L, self.rnn_hidden)

        for layer_idx in range(self.n_gru_layers):
            h_2d = GRULayer2D(hidden_size=self.rnn_hidden, name=f"gru2d_{layer_idx}")(x_2d)
            x_2d = x_2d + h_2d   # residual connection

        # Flatten back to (B, N, rnn_hidden) for the shared towers
        x = x_2d.reshape(B, N, self.rnn_hidden)

        return _amp_phase_towers(
            x,
            head_hidden=self.head_hidden,
            local_size=self.hilbert.local_size,
            machine_pow=self.machine_pow,
            phase_scale=self.phase_scale,
            phase_init_std=self.phase_init_std,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_model(cfg: Args, hilbert) -> nk.models.AbstractARNN:
    common = dict(
        hilbert=hilbert,
        L=cfg.L,
        embed_dim=cfg.embed_dim,
        rnn_hidden=cfg.rnn_hidden,
        head_hidden=cfg.head_hidden,
        n_gru_layers=cfg.n_gru_layers,
        machine_pow=cfg.machine_pow,
        phase_scale=cfg.phase_scale,
        phase_init_std=cfg.phase_init_std,
    )
    if cfg.model_type == "gru1d":
        return RecurrentAR2DEmbedding(**common)
    elif cfg.model_type == "gru2d":
        return RecurrentAR2DGRU(**common)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model_type!r}. Choose 'gru1d' or 'gru2d'.")


# ---------------------------------------------------------------------------
# Local energy
# ---------------------------------------------------------------------------

def make_local_energy_fn(model: nn.Module, H):
    def local_energy(params, configs):
        configs = configs.astype(jnp.int32)
        x_primes, mels = H.get_conn_padded(configs)   # (B, K, N), (B, K)
        B, K, N = x_primes.shape
        logpsi_x  = model.apply({"params": params}, configs)
        logpsi_xp = model.apply(
            {"params": params},
            x_primes.reshape(B * K, N),
        ).reshape(B, K)
        ratios = jnp.exp(logpsi_xp - logpsi_x[:, None])
        e_loc  = jnp.sum(mels * ratios, axis=1)
        return e_loc
    return local_energy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.3", config_name="config")
def main(cfg: Args) -> None:
    print(OmegaConf.to_yaml(cfg))
    print(jax.default_backend())

    N = cfg.L * cfg.L

    ckpt_dir = Path(cfg.output_directory) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpointer = ocp.StandardCheckpointer()

    if cfg.wandb_mode != "disabled":
        Path(cfg.wandb_directory).mkdir(parents=True, exist_ok=True)
        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=cfg.wandb_run_name,
            config=OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),
            dir=cfg.wandb_directory,
            mode=cfg.wandb_mode,
        )

    hi = nk.hilbert.Spin(s=1 / 2, N=N)

    # 2-D square lattice with NN (color 0) and NNN (color 1) edges
    lattice = nk.graph.Hypercube(length=cfg.L, n_dim=2, pbc=cfg.pbc, max_neighbor_order=2)
    H_nk = nk.operator.Heisenberg(
        hilbert=hi,
        graph=lattice,
        J=[cfg.J1, cfg.J2],
        sign_rule=[False, False],
    )
    H = H_nk.to_jax_operator() if hasattr(H_nk, "to_jax_operator") else H_nk

    model = make_model(cfg, hi)

    sampler = nk.sampler.ARDirectSampler(hilbert=hi)

    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=cfg.n_samples,
        seed=cfg.seed,
        n_discard_per_chain=0,
    )

    eval_vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=cfg.eval_batch_size,
        seed=cfg.seed + 1,
        n_discard_per_chain=cfg.eval_n_discard_per_chain,
    )

    params0 = vstate.variables["params"]
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params0))
    print(f"Model: {cfg.model_type}  N={N}  Parameters: {n_params:,}")

    sampler_state0 = vstate.sampler_state
    chain_length   = vstate.chain_length

    tx, lr_schedule = make_tx(cfg)
    state = TrainState.create(apply_fn=model.apply, params=params0, tx=tx)

    local_energy = make_local_energy_fn(model, H)

    # ------------------------------------------------------------------
    # Batch collection
    # ------------------------------------------------------------------

    def collect_batch(old_params, sampler_state):
        configs, sampler_state = sampler.sample(
            model,
            {"params": old_params},
            state=sampler_state,
            chain_length=chain_length,
            return_log_probabilities=False,
        )
        configs = jax.lax.stop_gradient(configs)
        configs_flat = configs.reshape(-1, N)

        old_logpsi = model.apply({"params": old_params}, configs_flat)
        old_logp   = cfg.machine_pow * jnp.real(old_logpsi)
        old_phase  = jnp.imag(old_logpsi)
        old_logp   = jax.lax.stop_gradient(old_logp)
        old_phase  = jax.lax.stop_gradient(old_phase)

        e_loc = local_energy(old_params, configs_flat)
        e_loc = jax.lax.stop_gradient(e_loc)

        return (configs_flat, old_logp, old_phase, e_loc), sampler_state

    # ------------------------------------------------------------------
    # Loss / metrics
    # ------------------------------------------------------------------

    def loss_terms_on_batch(params, batch):
        configs_flat, old_logp, old_phase, e_loc = batch

        new_logpsi = model.apply({"params": params}, configs_flat)
        new_logp   = cfg.machine_pow * jnp.real(new_logpsi)
        new_phase  = jnp.imag(new_logpsi)

        # ===== Amplitude channel =====
        e_real   = jnp.real(e_loc)
        adv_real = e_real - jnp.mean(e_real)
        if cfg.normalize_advantage:
            adv_real = adv_real / (jnp.std(adv_real) + 1e-8)

        log_ratio        = jnp.clip(new_logp - old_logp, -20.0, 20.0)
        ratio_real       = jnp.exp(log_ratio)
        clipped_ratio_real = jnp.clip(ratio_real, 1.0 - cfg.ppo_clip_eps, 1.0 + cfg.ppo_clip_eps)

        loss_real = jnp.mean(jnp.maximum(ratio_real * adv_real, clipped_ratio_real * adv_real))

        # ===== Phase channel =====
        e_imag    = jnp.imag(e_loc)
        adv_phase = e_imag
        if cfg.center_imag_advantage:
            adv_phase = adv_phase - jnp.mean(adv_phase)
        if cfg.normalize_imag_advantage:
            adv_phase = adv_phase / (jnp.std(adv_phase) + 1e-8)

        phase_delta = jnp.atan2(
            jnp.sin(new_phase - old_phase),
            jnp.cos(new_phase - old_phase),
        )
        is_ratio = jax.lax.stop_gradient(ratio_real)

        if cfg.phase_loss_type == "ratio":
            denom = jnp.where(
                jnp.abs(old_phase) >= cfg.phase_ratio_tau,
                old_phase,
                jnp.where(old_phase >= 0.0, cfg.phase_ratio_tau, -cfg.phase_ratio_tau),
            )
            ratio_phase         = new_phase / denom
            clipped_ratio_phase = jnp.clip(ratio_phase, 1.0 - cfg.phase_clip_eps, 1.0 + cfg.phase_clip_eps)
            loss_phase = jnp.mean(jnp.maximum(
                is_ratio * ratio_phase * adv_phase,
                is_ratio * clipped_ratio_phase * adv_phase,
            ))
            phase_stat = ratio_phase

        elif cfg.phase_loss_type == "delta_clip":
            clipped_phase_delta = jnp.clip(phase_delta, -cfg.phase_delta_clip, cfg.phase_delta_clip)
            loss_phase = jnp.mean(jnp.maximum(
                is_ratio * phase_delta * adv_phase,
                is_ratio * clipped_phase_delta * adv_phase,
            ))
            if cfg.phase_delta_l2_coef > 0.0:
                loss_phase = loss_phase + cfg.phase_delta_l2_coef * jnp.mean(phase_delta ** 2)
            phase_stat = phase_delta
        else:
            raise ValueError(f"Unknown phase_loss_type: {cfg.phase_loss_type}")

        losses = {"real": loss_real, "phase": loss_phase}
        total_loss = loss_real + cfg.phase_coef * loss_phase

        metrics = {
            "loss":               total_loss,
            "loss_real":          loss_real,
            "loss_phase":         loss_phase,
            "E_mean_real":        jnp.mean(e_real),
            "E_std_real":         jnp.std(e_real),
            "E_mean_imag":        jnp.mean(e_imag),
            "E_std_imag":         jnp.std(e_imag),
            "ratio_mean":         jnp.mean(ratio_real),
            "clip_frac":          jnp.mean(jnp.abs(ratio_real - 1.0) > cfg.ppo_clip_eps),
            "approx_kl":          jnp.mean(old_logp - new_logp),
            "phase_old_mean":     jnp.mean(old_phase),
            "phase_new_mean":     jnp.mean(new_phase),
            "phase_delta_mean":   jnp.mean(phase_delta),
            "phase_delta_rms":    jnp.sqrt(jnp.mean(phase_delta ** 2)),
            "phase_stat_mean":    jnp.mean(phase_stat),
            "phase_stat_abs_mean": jnp.mean(jnp.abs(phase_stat)),
        }

        return losses, metrics

    def phase_backward_weights(params, batch):
        configs_flat, _, old_phase, e_loc = batch
        new_logpsi = model.apply({"params": params}, configs_flat)
        new_phase  = jnp.imag(new_logpsi)

        e_imag    = jnp.imag(e_loc)
        adv_phase = e_imag
        if cfg.center_imag_advantage:
            adv_phase = adv_phase - jnp.mean(adv_phase)
        if cfg.normalize_imag_advantage:
            adv_phase = adv_phase / (jnp.std(adv_phase) + 1e-8)

        B = adv_phase.shape[0]

        if cfg.phase_loss_type == "delta_clip":
            phase_delta = jnp.atan2(
                jnp.sin(new_phase - old_phase),
                jnp.cos(new_phase - old_phase),
            )
            clipped = jnp.clip(phase_delta, -cfg.phase_delta_clip, cfg.phase_delta_clip)
            active  = (phase_delta * adv_phase >= clipped * adv_phase).astype(new_phase.dtype)
            w = active * adv_phase / B
            if cfg.phase_delta_l2_coef > 0.0:
                w = w + (2.0 * cfg.phase_delta_l2_coef / B) * phase_delta

        elif cfg.phase_loss_type == "ratio":
            denom = jnp.where(
                jnp.abs(old_phase) >= cfg.phase_ratio_tau,
                old_phase,
                jnp.where(old_phase >= 0.0, cfg.phase_ratio_tau, -cfg.phase_ratio_tau),
            )
            ratio_phase         = new_phase / denom
            clipped_ratio_phase = jnp.clip(ratio_phase, 1.0 - cfg.phase_clip_eps, 1.0 + cfg.phase_clip_eps)
            active = (ratio_phase * adv_phase >= clipped_ratio_phase * adv_phase).astype(new_phase.dtype)
            w = active * (adv_phase / denom) / B
        else:
            raise ValueError(f"Unknown phase_loss_type: {cfg.phase_loss_type}")

        return jax.lax.stop_gradient(w.astype(new_phase.dtype))

    def phase_linearized_objective(params, batch, weights):
        configs_flat, _, _, _ = batch
        new_logpsi = model.apply({"params": params}, configs_flat)
        new_phase  = jnp.imag(new_logpsi)
        return jnp.sum(jax.lax.stop_gradient(weights) * new_phase)

    def apply_phase_jacobian_baseline(params, batch, grads_imag):
        """
        Applies the variance-reducing baseline:
            g <- g - (sum_i w_i) * c*
        where
            c* = (sum_i w_i^2 O_i) / (sum_i w_i^2)
        """
        w    = phase_backward_weights(params, batch)
        w_sq = jax.lax.stop_gradient(w ** 2)

        c_num = jax.grad(phase_linearized_objective)(params, batch, w_sq)
        c_den = jnp.sum(w_sq) + cfg.phase_jacobian_baseline_eps
        c_tree = jax.tree_util.tree_map(
            lambda x, g: jax.lax.stop_gradient((x / c_den).astype(g.dtype)),
            c_num,
            grads_imag,
        )

        sum_w = jax.lax.stop_gradient(jnp.sum(w))
        grads_imag = tree_add(grads_imag, tree_scale(c_tree, -sum_w))
        return grads_imag

    def phase_objective(params, batch):
        w = phase_backward_weights(params, batch)
        return phase_linearized_objective(params, batch, w)

    def real_objective(params, batch):
        losses, _ = loss_terms_on_batch(params, batch)
        return losses["real"]

    # ------------------------------------------------------------------
    # Eval
    # ------------------------------------------------------------------

    @jax.jit
    def eval_chunk_sums(params, configs):
        e_loc     = local_energy(params, configs)
        e_real    = jnp.real(e_loc)
        e_imag    = jnp.imag(e_loc)
        return (
            jnp.sum(e_real),
            jnp.sum(e_real ** 2),
            jnp.sum(e_imag),
            jnp.sum(jnp.abs(e_imag)),
            e_real.shape[0],
        )

    def evaluate(state):
        eval_vstate.parameters = state.params

        total_real = total_real_sq = total_imag = total_abs_imag = 0.0
        total_n    = 0
        n_remaining = cfg.eval_n_samples

        while n_remaining > 0:
            eval_vstate.sample()
            configs = jnp.asarray(eval_vstate.samples).reshape(-1, N)
            cur_n   = min(configs.shape[0], n_remaining)
            configs = configs[:cur_n]

            real_sum, real_sq_sum, imag_sum, abs_imag_sum, chunk_n = eval_chunk_sums(
                state.params, configs
            )
            total_real      += float(real_sum)
            total_real_sq   += float(real_sq_sum)
            total_imag      += float(imag_sum)
            total_abs_imag  += float(abs_imag_sum)
            total_n         += int(chunk_n)
            n_remaining     -= int(chunk_n)

        energy_real  = total_real / total_n
        variance_real = max(total_real_sq / total_n - energy_real ** 2, 0.0)
        energy_imag  = total_imag / total_n
        abs_imag_mean = total_abs_imag / total_n
        denom        = max(energy_real * energy_real, 1e-12)
        v_score      = float(N * variance_real / denom)

        return {
            "eval_E_mean_real":       energy_real,
            "eval_E_mean_imag":       energy_imag,
            "eval_E_var_real":        variance_real,
            "eval_abs_E_imag_mean":   abs_imag_mean,
            "eval_V_score":           v_score,
            "eval_n_samples_used":    total_n,
        }

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    @partial(jax.jit, donate_argnums=(0, 1))
    def train_iter(state, sampler_state):
        old_params = state.params
        batch, sampler_state = collect_batch(old_params, sampler_state)

        def total_objective(params, batch):
            losses, metrics = loss_terms_on_batch(params, batch)
            total_loss = losses["real"] + cfg.phase_coef * losses["phase"]
            return total_loss, (losses, metrics)

        def epoch_step(state, _):
            (loss, (losses, metrics)), grads = jax.value_and_grad(
                total_objective, argnums=0, has_aux=True
            )(state.params, batch)
            # losses, metrics = loss_terms_on_batch(state.params, batch)

            # grads_real = jax.grad(real_objective)(state.params, batch)
            # grads_imag = jax.grad(phase_objective)(state.params, batch)

            # if cfg.use_phase_jacobian_baseline:
            #     grads_imag = apply_phase_jacobian_baseline(state.params, batch, grads_imag)

            # grads = tree_add(
            #     grads_real,
            #     tree_scale(grads_imag, cfg.phase_coef),
            # )

            # grad_norm_real = optax.global_norm(grads_real)
            # grad_norm_imag = optax.global_norm(grads_imag)
            grad_norm = optax.global_norm(grads)
            state = state.apply_gradients(grads=grads)

            metrics = {
                **metrics,
                # "grad_norm_real": grad_norm_real,
                # "grad_norm_imag": grad_norm_imag,
                "grad_norm": grad_norm,
                "param_norm": optax.global_norm(state.params),
            }
            return state, metrics

        state, metrics_hist = jax.lax.scan(epoch_step, state, xs=None, length=cfg.ppo_epochs)
        metrics = jax.tree_util.tree_map(lambda x: x[-1], metrics_hist)
        return state, sampler_state, metrics, batch

    sampler_state = sampler_state0

    for it in range(cfg.n_iter):
        state, sampler_state, metrics, batch = train_iter(state, sampler_state)

        if it % cfg.log_every == 0:
            current_lr = float(lr_schedule(state.step))
            msg = (
                f"it={it:04d}  loss={float(metrics['loss']): .6f}  "
                f"Lreal={float(metrics['loss_real']): .6f}  "
                f"Lphase={float(metrics['loss_phase']): .6f}  "
                f"E_real={float(metrics['E_mean_real']): .6f} ± {float(metrics['E_std_real']): .6f}  "
                f"E_imag={float(metrics['E_mean_imag']): .6f} ± {float(metrics['E_std_imag']): .6f}  "
                f"ratio={float(metrics['ratio_mean']): .6f}  "
                f"clip={float(metrics['clip_frac']): .3f}  "
                f"kl={float(metrics['approx_kl']): .3e}  "
                f"dphi_rms={float(metrics['phase_delta_rms']): .3e}  "
                f"||g||={float(metrics['grad_norm']): .3e}  "
                f"||θ||={float(metrics['param_norm']): .3e}"
            )
            print(msg)

            if cfg.wandb_mode != "disabled":
                wandb.log({
                    "iter":                 it,
                    "lr":                   current_lr,
                    "loss":                 float(metrics["loss"]),
                    "loss_real":            float(metrics["loss_real"]),
                    "loss_phase":           float(metrics["loss_phase"]),
                    "E_mean_real":          float(metrics["E_mean_real"]),
                    "E_std_real":           float(metrics["E_std_real"]),
                    "E_mean_imag":          float(metrics["E_mean_imag"]),
                    "E_std_imag":           float(metrics["E_std_imag"]),
                    "grad_norm":            float(metrics["grad_norm"]),
                    "param_norm":           float(metrics["param_norm"]),
                    "ratio_mean":           float(metrics["ratio_mean"]),
                    "clip_frac":            float(metrics["clip_frac"]),
                    "approx_kl":            float(metrics["approx_kl"]),
                    "phase_old_mean":       float(metrics["phase_old_mean"]),
                    "phase_new_mean":       float(metrics["phase_new_mean"]),
                    "phase_delta_mean":     float(metrics["phase_delta_mean"]),
                    "phase_delta_rms":      float(metrics["phase_delta_rms"]),
                    "phase_stat_mean":      float(metrics["phase_stat_mean"]),
                    "phase_stat_abs_mean":  float(metrics["phase_stat_abs_mean"]),
                }, step=it)

        if cfg.save_checkpoint_every > 0 and it > 0 and it % cfg.save_checkpoint_every == 0:
            checkpointer.save(
                ckpt_dir / f"iter_{it:07d}",
                args=ocp.args.StandardSave(state.params),
            )

        if cfg.eval_every > 0 and (it % cfg.eval_every == 0 or it == cfg.n_iter - 1):
            eval_metrics = evaluate(state)

            if cfg.log_gradient_info:
                grads_real  = jax.grad(real_objective)(state.params, batch)
                grads_phase = jax.grad(phase_objective)(state.params, batch)
                eval_metrics.update({
                    "eval_grad_norm_real":  float(optax.global_norm(grads_real)),
                    "eval_grad_norm_phase": float(optax.global_norm(grads_phase)),
                })

            print(
                f"[eval] it={it:04d}  "
                f"E_real={eval_metrics['eval_E_mean_real']:.8f}  "
                f"E_imag={eval_metrics['eval_E_mean_imag']:.8e}  "
                f"Var_real={eval_metrics['eval_E_var_real']:.8e}  "
                f"|E_imag|={eval_metrics['eval_abs_E_imag_mean']:.8e}  "
                f"V_score={eval_metrics['eval_V_score']:.8e}"
            )

            if cfg.wandb_mode != "disabled":
                wandb.log(eval_metrics, step=it)

    # Save final checkpoint
    final_ckpt_path = ckpt_dir / "final"
    checkpointer.save(
        final_ckpt_path,
        args=ocp.args.StandardSave(state.params),
    )
    print(f"Saved final checkpoint to {final_ckpt_path}")

    if cfg.wandb_mode != "disabled":
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description=f"Final params: {cfg.model_type}, L={cfg.L}, J2/J1={cfg.J2/cfg.J1:.3f}",
        )
        artifact.add_dir(str(final_ckpt_path))
        wandb.log_artifact(artifact)

    vstate.parameters = state.params
    vstate.sample()
    final_stats = vstate.expect(H_nk)
    final_energy = final_stats.mean.real
    final_energy_err = final_stats.error_of_mean.real
    final_variance = final_stats.variance.real
    print(
        f"Final energy (NetKet expect): {final_energy:.10f} ± {final_energy_err:.2e}  "
        f"(per site: {final_energy / N:.10f})"
    )

    if cfg.wandb_mode != "disabled":
        wandb.log({
            "final_E":           final_energy,
            "final_E_per_site":  final_energy / N,
            "final_E_err":       final_energy_err,
            "final_E_variance":  final_variance,
        })
        wandb.finish()


if __name__ == "__main__":
    main()
