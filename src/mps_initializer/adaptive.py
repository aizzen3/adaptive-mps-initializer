import numpy as np

from .state import QuantumState
from .mps import statevector_to_mps
from .metrics import state_fidelity
from .compression import find_min_bond_dim_for_fidelity
from .circuits import (
    mps_to_sequential_qiskit_circuit,
    physical_statevector_from_mps_circuit,
)
from .circuit_metrics import circuit_gate_report


def adaptive_mps_initializer(
    data,
    fidelity_threshold=0.999,
    max_search_dim=64,
    cutoff=1e-12,
    normalize=True,
    decompose_gates=True,
):
    """
    Adaptive MPS state-preparation pipeline.

    This function:
        1. normalizes the target statevector
        2. finds the smallest bond dimension D reaching target fidelity
        3. builds a compressed MPS using that D
        4. builds a sequential Qiskit MPS circuit
        5. checks preparation fidelity
        6. returns circuit + report
    """

    state = QuantumState(data, normalize=normalize)
    psi = state.data

    best, sweep = find_min_bond_dim_for_fidelity(
        psi,
        threshold=fidelity_threshold,
        max_search_dim=max_search_dim,
        cutoff=cutoff,
        normalize=False,
    )

    if best is None:
        raise ValueError(
            f"No bond dimension up to {max_search_dim} reached "
            f"fidelity threshold {fidelity_threshold}."
        )

    D = best["max_bond_dim_allowed"]

    compressed_mps = statevector_to_mps(
        psi,
        max_bond_dim=D,
        cutoff=cutoff,
        normalize=False,
    )

    qc = mps_to_sequential_qiskit_circuit(compressed_mps)

    prepared, leakage = physical_statevector_from_mps_circuit(
        qc,
        n_physical=state.n_qubits,
    )

    prep_fidelity = state_fidelity(psi, prepared)

    gate_report = circuit_gate_report(
        qc,
        decompose=decompose_gates,
        reps=10,
    )

    report = {
        "n_qubits": state.n_qubits,
        "statevector_length": len(psi),
        "target_fidelity_threshold": fidelity_threshold,
        "chosen_max_bond_dim": D,
        "actual_bond_dimensions": compressed_mps.bond_dimensions(),
        "actual_max_bond_dim": compressed_mps.max_bond_dimension(),
        "bond_qubits": int(np.ceil(np.log2(compressed_mps.max_bond_dimension())))
        if compressed_mps.max_bond_dimension() > 1
        else 0,
        "total_circuit_qubits": qc.num_qubits,
        "compression_fidelity": best["fidelity"],
        "circuit_preparation_fidelity": prep_fidelity,
        "bond_leakage": leakage,
        "compression_sweep": sweep,
        "gate_report": gate_report,
    }

    return qc, compressed_mps, report
