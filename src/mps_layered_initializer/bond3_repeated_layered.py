"""
Repeated projected D=3 layered MPS initializer.

Each layer:
    current physical residual
    -> D=3 ancilla MPS layer
    -> apply inverse layer
    -> project bond qubits back to |00>
    -> normalize
    -> repeat

This is the practical D=3 layered version on qubit hardware.
"""

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from mps_initializer.builders import statevector_from_function
from mps_initializer.metrics import normalize_statevector, state_fidelity
from mps_initializer.circuits import physical_statevector_from_mps_circuit
from mps_initializer.circuit_metrics import circuit_gate_report

from .bond3_ancilla_layer import bond3_ancilla_layer_from_statevector


def _embed_physical_with_zero_bond(physical_state, n_physical, n_bond):
    """
    Embed |psi>_physical into |psi>_physical |00>_bond.
    """
    physical_state = np.asarray(physical_state, dtype=complex)

    physical_dim = 2**n_physical
    total_dim = 2 ** (n_physical + n_bond)

    full_state = np.zeros(total_dim, dtype=complex)
    full_state[:physical_dim] = physical_state

    return full_state


def _project_zero_bond(full_state, n_physical):
    """
    Extract the bond-|00> sector and normalize it.
    """
    physical_dim = 2**n_physical

    projected = np.asarray(
        full_state[:physical_dim],
        dtype=complex,
    )

    norm = float(np.linalg.norm(projected))

    if norm > 0:
        projected = projected / norm

    return projected, norm


def _lift_layer(layer, total_qubits):
    """
    Make sure every layer acts on the same total number of qubits.
    """
    if layer.num_qubits == total_qubits:
        return layer

    if layer.num_qubits > total_qubits:
        raise ValueError("Layer has too many qubits.")

    lifted = QuantumCircuit(total_qubits)

    lifted.compose(
        layer,
        qubits=list(range(layer.num_qubits)),
        inplace=True,
    )

    return lifted


def _build_preparation_circuit(layers, total_qubits):
    """
    If residual update is:

        current <- U_layer^dagger current

    then preparation is:

        U_1 U_2 ... U_L |0>

    In Qiskit append order, we compose reversed layers.
    """
    circuit = QuantumCircuit(total_qubits)

    for layer in reversed(layers):
        circuit.compose(layer, inplace=True)

    return circuit


def repeated_bond3_layers_from_statevector(
    psi,
    fidelity_threshold=0.999,
    max_layers=10,
    cutoff=1e-12,
    normalize=True,
    decompose_gates=True,
):
    """
    Repeated projected D=3 layers from a statevector.
    """
    target = np.asarray(psi, dtype=complex)

    if normalize:
        target = normalize_statevector(target)

    n_physical = int(np.log2(len(target)))

    if len(target) != 2**n_physical:
        raise ValueError("Statevector length must be a power of 2.")

    n_bond = 2
    total_qubits = n_physical + n_bond

    current_physical = target.copy()

    layers = []
    history = []

    best_layers = []
    best_fidelity = -1.0
    best_prepared = None
    hit_target = False

    for layer_number in range(1, max_layers + 1):

        layer_result = bond3_ancilla_layer_from_statevector(
            current_physical,
            cutoff=cutoff,
            normalize=True,
            decompose_gates=False,
        )

        layer = _lift_layer(
            layer_result["circuit"],
            total_qubits=total_qubits,
        )

        layers.append(layer)

        full_current = _embed_physical_with_zero_bond(
            current_physical,
            n_physical=n_physical,
            n_bond=n_bond,
        )

        U = Operator(layer).data

        full_residual = U.conj().T @ full_current
        full_residual = full_residual / np.linalg.norm(full_residual)

        current_physical, projection_norm = _project_zero_bond(
            full_residual,
            n_physical=n_physical,
        )

        prep_circuit = _build_preparation_circuit(
            layers,
            total_qubits=total_qubits,
        )

        prepared_physical, bond_leakage = physical_statevector_from_mps_circuit(
            prep_circuit,
            n_physical=n_physical,
            normalize=True,
        )

        fidelity = state_fidelity(
            target,
            prepared_physical,
            normalize=True,
        )

        residual_overlap_zero = float(
            abs(current_physical[0]) ** 2
        )

        row = {
            "layer": layer_number,
            "fidelity": fidelity,
            "projection_norm": projection_norm,
            "bond_leakage": bond_leakage,
            "residual_overlap_zero": residual_overlap_zero,
            "layer_bonds": layer_result["report"]["actual_bond_dimensions"],
            "layer_total_qubits": layer.num_qubits,
        }

        history.append(row)

        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_layers = list(layers)
            best_prepared = prepared_physical

        if fidelity >= fidelity_threshold:
            hit_target = True
            break

    chosen_layers = list(layers) if hit_target else best_layers

    final_circuit = _build_preparation_circuit(
        chosen_layers,
        total_qubits=total_qubits,
    )

    final_prepared, final_bond_leakage = physical_statevector_from_mps_circuit(
        final_circuit,
        n_physical=n_physical,
        normalize=True,
    )

    final_fidelity = state_fidelity(
        target,
        final_prepared,
        normalize=True,
    )

    gate_report = circuit_gate_report(
        final_circuit,
        decompose=decompose_gates,
    )

    report = {
        "method": "repeated_projected_bond3_layers",
        "bond_dim": 3,
        "n_physical_qubits": n_physical,
        "bond_qubits": n_bond,
        "total_circuit_qubits": total_qubits,
        "fidelity_threshold": fidelity_threshold,
        "max_layers": max_layers,
        "chosen_layers": len(chosen_layers),
        "hit_target": hit_target,
        "fidelity": final_fidelity,
        "best_fidelity": best_fidelity,
        "bond_leakage": final_bond_leakage,
        "history": history,
        "gate_report": gate_report,
        "note": (
            "Projected D=3 repeated layers: after each inverse layer, "
            "the residual is projected back to the bond-|00> physical sector."
        ),
    }

    return {
        "circuit": final_circuit,
        "report": report,
        "history": history,
        "prepared_statevector": final_prepared,
    }


def repeated_bond3_layers_from_function(
    func,
    n_qubits,
    x_min=0.0,
    x_max=1.0,
    mode="amplitude",
    fidelity_threshold=0.999,
    max_layers=10,
    cutoff=1e-12,
    decompose_gates=True,
):
    """
    Function-input version of repeated_bond3_layers_from_statevector.
    """
    psi, x, values = statevector_from_function(
        func,
        n_qubits=n_qubits,
        x_min=x_min,
        x_max=x_max,
        mode=mode,
    )

    result = repeated_bond3_layers_from_statevector(
        psi,
        fidelity_threshold=fidelity_threshold,
        max_layers=max_layers,
        cutoff=cutoff,
        normalize=False,
        decompose_gates=decompose_gates,
    )

    result["x"] = x
    result["values"] = values
    result["statevector"] = psi

    return result
