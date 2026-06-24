"""
Fixed-bond MPS layer tools.

This is the first step toward a bond-2 / bond-3 layered initializer.

For now:
    target state
    -> fixed-D MPS approximation
    -> Qiskit circuit
    -> fidelity report

Later:
    repeat this as multiple layers.
"""

import numpy as np

from mps_initializer.builders import statevector_from_function
from mps_initializer.mps import statevector_to_mps
from mps_initializer.metrics import normalize_statevector, state_fidelity
from mps_initializer.circuits import (
    mps_to_sequential_qiskit_circuit,
    physical_statevector_from_mps_circuit,
)
from mps_initializer.circuit_metrics import circuit_gate_report


def fixed_bond_mps_layer_from_statevector(
    psi,
    bond_dim=3,
    cutoff=1e-12,
    normalize=True,
    decompose_gates=True,
):
    target = np.asarray(psi, dtype=complex)

    if normalize:
        target = normalize_statevector(target)

    n_qubits = int(np.log2(len(target)))

    if len(target) != 2**n_qubits:
        raise ValueError("Statevector length must be a power of 2.")

    mps = statevector_to_mps(
        target,
        max_bond_dim=bond_dim,
        cutoff=cutoff,
        normalize=True,
    )

    qc = mps_to_sequential_qiskit_circuit(
        mps,
        label_prefix=f"D{bond_dim}_A",
    )

    prepared, leakage = physical_statevector_from_mps_circuit(
        qc,
        n_physical=n_qubits,
        normalize=True,
    )

    fidelity = state_fidelity(
        target,
        prepared,
        normalize=True,
    )

    gate_report = circuit_gate_report(
        qc,
        decompose=decompose_gates,
    )

    report = {
        "method": "fixed_bond_mps_layer",
        "bond_dim": bond_dim,
        "n_physical_qubits": n_qubits,
        "statevector_length": len(target),
        "actual_bond_dimensions": mps.bond_dimensions(),
        "actual_max_bond_dim": mps.max_bond_dimension(),
        "bond_qubits": int(np.ceil(np.log2(max(1, mps.max_bond_dimension())))),
        "total_circuit_qubits": qc.num_qubits,
        "fidelity": fidelity,
        "bond_leakage": leakage,
        "gate_report": gate_report,
    }

    return {
        "circuit": qc,
        "mps": mps,
        "prepared_statevector": prepared,
        "report": report,
    }


def fixed_bond_mps_layer_from_function(
    func,
    n_qubits,
    x_min=0.0,
    x_max=1.0,
    mode="amplitude",
    bond_dim=3,
    cutoff=1e-12,
    decompose_gates=True,
):
    psi, x, values = statevector_from_function(
        func,
        n_qubits=n_qubits,
        x_min=x_min,
        x_max=x_max,
        mode=mode,
    )

    result = fixed_bond_mps_layer_from_statevector(
        psi,
        bond_dim=bond_dim,
        cutoff=cutoff,
        normalize=False,
        decompose_gates=decompose_gates,
    )

    result["x"] = x
    result["values"] = values
    result["statevector"] = psi

    return result
