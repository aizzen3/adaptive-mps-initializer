import numpy as np

from .state import QuantumState
from .mps import statevector_to_mps
from .metrics import state_fidelity
from .diagnostics import (
    entanglement_entropy_profile,
    rank_profile,
    cumulative_rank_profile,
)
from .compression import find_min_bond_dim_for_fidelity


def analyze_statevector(
    data,
    fidelity_threshold=0.999,
    max_search_dim=64,
    cumulative_weight=0.999999,
    normalize=True,
):
    """
    Analyze a quantum statevector using MPS diagnostics.

    Returns:
        dictionary containing:
            number of qubits
            exact bond dimensions
            entropy profile
            rank profile
            cumulative rank profile
            minimum bond dimension for target fidelity
    """

    state = QuantumState(data, normalize=normalize)
    psi = state.data

    exact_mps = statevector_to_mps(psi, normalize=False)
    reconstructed = exact_mps.to_statevector()

    exact_fidelity = state_fidelity(psi, reconstructed)

    best, sweep = find_min_bond_dim_for_fidelity(
        psi,
        threshold=fidelity_threshold,
        max_search_dim=max_search_dim,
        normalize=False,
    )

    report = {
        "n_qubits": state.n_qubits,
        "statevector_length": len(psi),
        "norm": float(np.linalg.norm(psi)),
        "exact_bond_dimensions": exact_mps.bond_dimensions(),
        "exact_max_bond_dimension": exact_mps.max_bond_dimension(),
        "exact_fidelity": exact_fidelity,
        "entropy_profile": entanglement_entropy_profile(psi).tolist(),
        "rank_profile": rank_profile(psi).tolist(),
        "cumulative_rank_profile": cumulative_rank_profile(
            psi,
            weight=cumulative_weight,
        ).tolist(),
        "fidelity_threshold": fidelity_threshold,
        "best_compressed_result": best,
        "compression_sweep": sweep,
    }

    return report
