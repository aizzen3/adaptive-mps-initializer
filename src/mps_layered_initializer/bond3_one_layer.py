"""
Bond-3 one-layer MPS initializer.

This is the correct first D=3 construction.

Important:
    D = 3 cannot be represented by a single virtual qubit.
    Therefore we embed the virtual bond dimension 3 into 2 bond qubits.

For n physical qubits:
    total circuit qubits = n + ceil(log2(3)) = n + 2

This module prepares one compressed D=3 MPS layer and extracts the
physical statevector after the bond qubits return to |0>.
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


def one_layer_bond3_from_statevector(
    psi,
    cutoff=1e-12,
    normalize=True,
    decompose_gates=True,
):
    """
    Prepare a state using one bond-3 MPS layer.

    Parameters
    ----------
    psi:
        Target statevector of length 2^n.

    cutoff:
        Singular-value cutoff used during MPS compression.

    normalize:
        If True, normalize input statevector.

    decompose_gates:
        If True, decompose generic unitaries into Qiskit basis gates
        for depth and CX counting.

    Returns
    -------
    dict with:
        circuit
        mps
        prepared_statevector
        report
    """

    target = np.asarray(psi, dtype=complex)

    if normalize:
        target = normalize_statevector(target)

    n_qubits = int(np.log2(len(target)))

    if len(target) != 2**n_qubits:
        raise ValueError("Statevector length must be a power of 2.")

    bond_dim = 3

    # Step 1: compress target into a bond-3 MPS
    mps = statevector_to_mps(
        target,
        max_bond_dim=bond_dim,
        cutoff=cutoff,
        normalize=True,
    )

    actual_bonds = mps.bond_dimensions()
    actual_max_bond = mps.max_bond_dimension()

    bond_qubits = int(np.ceil(np.log2(max(1, actual_max_bond))))

    # Step 2: convert bond-3 MPS into sequential MPS circuit
    # This circuit acts on:
    #   physical qubits + bond ancilla qubits
    circuit = mps_to_sequential_qiskit_circuit(
        mps,
        label_prefix="D3_A",
    )

    # Step 3: extract the physical statevector from the full circuit
    prepared, leakage = physical_statevector_from_mps_circuit(
        circuit,
        n_physical=n_qubits,
        normalize=True,
    )

    fidelity = state_fidelity(
        target,
        prepared,
        normalize=True,
    )

    gate_report = circuit_gate_report(
        circuit,
        decompose=decompose_gates,
    )

    report = {
        "method": "one_layer_bond3_mps",
        "bond_dim": bond_dim,
        "n_physical_qubits": n_qubits,
        "statevector_length": len(target),
        "actual_bond_dimensions": actual_bonds,
        "actual_max_bond_dim": actual_max_bond,
        "bond_qubits": bond_qubits,
        "total_circuit_qubits": circuit.num_qubits,
        "fidelity": fidelity,
        "bond_leakage": leakage,
        "gate_report": gate_report,
    }

    return {
        "circuit": circuit,
        "mps": mps,
        "prepared_statevector": prepared,
        "report": report,
    }


def one_layer_bond3_from_function(
    func,
    n_qubits,
    x_min=0.0,
    x_max=1.0,
    mode="amplitude",
    cutoff=1e-12,
    decompose_gates=True,
):
    """
    Function-input version of one_layer_bond3_from_statevector.
    """

    psi, x, values = statevector_from_function(
        func,
        n_qubits=n_qubits,
        x_min=x_min,
        x_max=x_max,
        mode=mode,
    )

    result = one_layer_bond3_from_statevector(
        psi,
        cutoff=cutoff,
        normalize=False,
        decompose_gates=decompose_gates,
    )

    result["x"] = x
    result["values"] = values
    result["statevector"] = psi

    return result
