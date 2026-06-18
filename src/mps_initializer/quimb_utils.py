import numpy as np

from .metrics import state_fidelity


def mps_to_quimb_arrays(mps):
    """
    Convert our MPS tensor arrays into Quimb-compatible arrays.

    Our tensor shape:
        (left_bond, physical_dim, right_bond)

    Quimb MPS tensor shape:
        (left_bond, right_bond, physical_dim)
    """

    return [
        np.transpose(A, (0, 2, 1))
        for A in mps.tensors
    ]


def mps_to_quimb_mps(mps):
    """
    Convert our MatrixProductState into quimb.tensor.MatrixProductState.
    """

    try:
        import quimb.tensor as qtn
    except ImportError as exc:
        raise ImportError("Quimb is required for mps_to_quimb_mps().") from exc

    arrays = mps_to_quimb_arrays(mps)

    return qtn.MatrixProductState(
        arrays,
        shape="lrp",
    )


def quimb_mps_to_statevector(qmps):
    """
    Convert Quimb MPS back to dense statevector.
    """

    dense = qmps.to_dense()

    return np.asarray(dense, dtype=complex).reshape(-1)


def compare_with_quimb(mps, target_statevector=None):
    """
    Compare our MPS contraction with Quimb contraction.

    If target_statevector is given, also compute fidelity with target.
    """

    our_state = mps.to_statevector()

    qmps = mps_to_quimb_mps(mps)
    quimb_state = quimb_mps_to_statevector(qmps)

    report = {
        "our_state_norm": float(np.linalg.norm(our_state)),
        "quimb_state_norm": float(np.linalg.norm(quimb_state)),
        "our_vs_quimb_fidelity": state_fidelity(our_state, quimb_state),
        "quimb_bond_sizes": list(qmps.bond_sizes()),
    }

    if target_statevector is not None:
        report["target_vs_our_fidelity"] = state_fidelity(
            target_statevector,
            our_state,
        )
        report["target_vs_quimb_fidelity"] = state_fidelity(
            target_statevector,
            quimb_state,
        )

    return report


def quimb_compression_sweep(mps, target_statevector, max_bond_dims, cutoff=1e-12):
    """
    Compress our MPS using Quimb and compare fidelity with target statevector.

    This is useful as an independent tensor-network check.
    """

    results = []

    for D in max_bond_dims:
        qmps = mps_to_quimb_mps(mps)
        qmps_c = qmps.copy()

        # Quimb compression is in-place.
        try:
            qmps_c.compress(max_bond=int(D), cutoff=cutoff)
        except TypeError:
            # Some versions accept fewer keyword arguments.
            qmps_c.compress(max_bond=int(D))

        compressed_state = quimb_mps_to_statevector(qmps_c)

        results.append(
            {
                "max_bond_dim_allowed": int(D),
                "quimb_bond_sizes": list(qmps_c.bond_sizes()),
                "quimb_max_bond": int(max(qmps_c.bond_sizes())) if len(qmps_c.bond_sizes()) > 0 else 1,
                "fidelity": state_fidelity(target_statevector, compressed_state),
                "compressed_norm": float(np.linalg.norm(compressed_state)),
            }
        )

    return results
