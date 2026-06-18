import numpy as np

from .state import QuantumState
from .mps import statevector_to_mps
from .metrics import state_fidelity


def compression_sweep(data, max_bond_dims, cutoff=1e-12, normalize=True):
    """
    Test different maximum bond dimensions and measure fidelity.

    Parameters
    ----------
    data:
        Input statevector.

    max_bond_dims:
        List of bond dimensions to test, for example:
            [1, 2, 4, 8, 16]

    cutoff:
        Singular-value cutoff.

    normalize:
        Normalize input statevector before compression.

    Returns
    -------
    results:
        List of dictionaries.
    """

    target = QuantumState(data, normalize=normalize).data

    results = []

    for D in max_bond_dims:
        mps = statevector_to_mps(
            target,
            max_bond_dim=int(D),
            cutoff=cutoff,
            normalize=False,
        )

        reconstructed = mps.to_statevector()

        fid = state_fidelity(
            target,
            reconstructed,
            normalize=True,
        )

        results.append(
            {
                "max_bond_dim_allowed": int(D),
                "actual_max_bond_dim": mps.max_bond_dimension(),
                "bond_dimensions": mps.bond_dimensions(),
                "fidelity": fid,
                "reconstructed_norm": float(np.linalg.norm(reconstructed)),
            }
        )

    return results


def find_min_bond_dim_for_fidelity(
    data,
    threshold=0.999,
    max_search_dim=64,
    cutoff=1e-12,
    normalize=True,
):
    """
    Find the smallest max_bond_dim that reaches a target fidelity.
    """

    dims = list(range(1, max_search_dim + 1))

    results = compression_sweep(
        data,
        max_bond_dims=dims,
        cutoff=cutoff,
        normalize=normalize,
    )

    for row in results:
        if row["fidelity"] >= threshold:
            return row, results

    return None, results
