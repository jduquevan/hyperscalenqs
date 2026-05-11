"""
Count the number of parameters for each model size used in the scaling experiment.

Usage:
    python analysis/count_parameters.py

Requires JAX, Flax, and NetKet to be installed (same environment as the training scripts).
"""

from __future__ import annotations

import os
import sys
import math
from pathlib import Path

# Override before any JAX import — the login node / CPU-only machines have no GPU
os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Allow importing from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import jax
import jax.numpy as jnp
import netket as nk
from qps import ComplexRecurrentAR

N_SITES = 12
hilbert = nk.hilbert.Spin(s=0.5, N=N_SITES)

# ---------------------------------------------------------------------------
# Size configurations: (label, n_gru_layers, rnn_hidden, head_hidden)
# ---------------------------------------------------------------------------
SIZES = [
    ("Tiny",   1,  64,  64),
    ("Small",  2, 128, 128),
    ("Medium", 3, 256, 256),
]

# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_params(model: ComplexRecurrentAR, N: int) -> int:
    """Initialise the model with a dummy batch and count all parameters."""
    key = jax.random.PRNGKey(0)
    # The model expects integer spin configs in {-1, +1}, shape (B, N)
    dummy_input = jnp.ones((1, N), dtype=jnp.float32)
    variables = model.init(key, dummy_input)
    leaves = jax.tree_util.tree_leaves(variables["params"])
    return int(sum(x.size for x in leaves))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    header = f"{'Size':<8}  {'GRU layers':>10}  {'RNN hidden':>10}  {'Head hidden':>11}  {'Parameters':>12}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for label, n_gru_layers, rnn_hidden, head_hidden in SIZES:
        model = ComplexRecurrentAR(
            hilbert=hilbert,
            embed_dim=32,
            rnn_hidden=rnn_hidden,
            head_hidden=head_hidden,
            n_gru_layers=n_gru_layers,
            use_site_embedding=True,
        )
        n_params = count_params(model, N_SITES)
        print(
            f"{label:<8}  {n_gru_layers:>10}  {rnn_hidden:>10}  {head_hidden:>11}  {n_params:>12,}"
        )

    print(sep)


if __name__ == "__main__":
    main()
