import numpy as np

from mps_initializer.builders import statevector_from_function
from mps_initializer.mps import statevector_to_mps
from mps_initializer.metrics import normalize_statevector, state_fidelity
from mps_initializer.circuits import (
    mps_to_sequential_qiskit_circuit,
    physical_statevector_from_mps_circuit,
)
from mps_initializer.circuit_metrics import circuit_gate_report

from mps_initializer.builders import statevector_from_function


def _local_block_summary(circuit):
    """
    Count how many local unitary blocks act on 1, 2, 3, ... qubits.

    This is before decomposition.
    """
    summary = {}

    for instruction in circuit.data:
        operation = instruction.operation
        qubits = instruction.qubits

        name = operation.name
        width = len(qubits)

        key = f"{width}_qubit_blocks"

        if name == "unitary":
            summary[key] = summary.get(key, 0) + 1

    return summary


def bond3_ancilla_layer_from_statevector(
    psi,
    cutoff=1e-12,
    normalize=True,
    decompose_gates=True,
):
    """
    Build one correct D=3 ancilla G-layer from a statevector.

    This does:
        target state
        -> bond-3 MPS approximation
        -> sequential ancilla G-layer circuit
        -> extract physical state
        -> compute fidelity

    The circuit uses:
        n physical qubits + 2 bond qubits
    """
    target = np.asarray(psi, dtype=complex)

    if normalize:
        target = normalize_statevector(target)

    n_physical = int(np.log2(len(target)))

    if len(target) != 2**n_physical:
        raise ValueError("Statevector length must be a power of 2.")

    bond_dim = 3

    mps = statevector_to_mps(
        target,
        max_bond_dim=bond_dim,
        cutoff=cutoff,
        normalize=True,
    )

    circuit = mps_to_sequential_qiskit_circuit(
        mps,
        label_prefix="D3G",
    )

    prepared_physical, bond_leakage = physical_statevector_from_mps_circuit(
        circuit,
        n_physical=n_physical,
        normalize=True,
    )

    fidelity = state_fidelity(
        target,
        prepared_physical,
        normalize=True,
    )

    gate_report = circuit_gate_report(
        circuit,
        decompose=decompose_gates,
    )

    local_blocks = _local_block_summary(circuit)

    report = {
        "method": "bond3_ancilla_G_layer",
        "bond_dim": bond_dim,
        "n_physical_qubits": n_physical,
        "bond_qubits": int(np.ceil(np.log2(mps.max_bond_dimension()))),
        "total_circuit_qubits": circuit.num_qubits,
        "actual_bond_dimensions": mps.bond_dimensions(),
        "actual_max_bond_dim": mps.max_bond_dimension(),
        "local_blocks_before_decomposition": local_blocks,
        "fidelity": fidelity,
        "bond_leakage": bond_leakage,
        "gate_report": gate_report,
        "note": (
            "Correct D=3 layer: local blocks act on one physical qubit "
            "and two bond ancilla qubits. Qiskit decomposes these blocks "
            "into one-qubit and two-qubit gates."
        ),
    }

    return {
        "circuit": circuit,
        "mps": mps,
        "prepared_statevector": prepared_physical,
        "report": report,
    }


def bond3_ancilla_layer_from_function(
    func,
    n_qubits,
    x_min=0.0,
    x_max=1.0,
    mode="amplitude",
    cutoff=1e-12,
    decompose_gates=True,
):
    """
    Function-input version of bond3_ancilla_layer_from_statevector.
    """
    psi, x, values = statevector_from_function(
        func,
        n_qubits=n_qubits,
        x_min=x_min,
        x_max=x_max,
        mode=mode,
    )

    result = bond3_ancilla_layer_from_statevector(
        psi,
        cutoff=cutoff,
        normalize=False,
        decompose_gates=decompose_gates,
    )

    result["x"] = x
    result["values"] = values
    result["statevector"] = psi

    return result
