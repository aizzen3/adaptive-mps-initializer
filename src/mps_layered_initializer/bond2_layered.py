"""
Old-style repeated bond-2 MPS layered initializer.

This is intentionally close to the BlackWild/qiskit-mps-initializer logic.

Goal:
    target state
    -> bond-2 MPS approximation
    -> G matrices
    -> one layer of 1- and 2-qubit gates
    -> apply G† to residual state
    -> repeat
    -> reverse layers to prepare the original state
"""

import numpy as np
import scipy.linalg
import quimb.gates as gates
import quimb.tensor as qtn

import qiskit
import qiskit.quantum_info

from mps_initializer.builders import statevector_from_function
from mps_initializer.metrics import normalize_statevector, state_fidelity
from mps_initializer.circuit_metrics import circuit_gate_report


def bond2_mps_approximation(psi):
    """
    Create the bond-2 MPS approximation of a normalized statevector.
    This follows the old package convention.
    """
    psi = np.asarray(psi, dtype=complex)

    if not np.isclose(np.linalg.norm(psi), 1.0):
        raise ValueError(
            "The state vector must be normalized. The norm was: "
            + str(np.linalg.norm(psi))
        )

    mps = qtn.MatrixProductState.from_dense(
        psi,
        max_bond=2,
        absorb="left",
    )

    mps.normalize()
    mps.right_canonicalize(inplace=True)
    mps.permute_arrays(shape="lpr")

    return mps


def G_matrices(mps):
    """
    Construct G matrices from a bond-2 MPS.

    This is copied closely from the old package logic.
    """
    max_bond = mps.max_bond()

    if max_bond is None:
        raise ValueError("This function does not support one-site MPS yet.")

    if max_bond > 2:
        raise ValueError("The maximum bond dimension of the MPS must be 2.")

    # Expand virtual bonds to dimension 2 for edge cases
    mps.expand_bond_dimension(2)

    G = []

    if mps.num_tensors > 1:
        tensor_data = mps[0].data.copy()

        A0_vec = tensor_data.reshape(4, 1)

        null_space = scipy.linalg.null_space(A0_vec.T)

        G0 = np.column_stack((A0_vec, null_space.conjugate()))

        G.append(G0)

        for i in range(1, mps.num_tensors - 1):
            Ai_a_0_data = mps[i].data[0, :, :].copy()
            Ai_a_0 = Ai_a_0_data.reshape(4, 1)

            Ai_a_1_data = mps[i].data[1, :, :].copy()
            Ai_a_1 = Ai_a_1_data.reshape(4, 1)

            if np.allclose(Ai_a_1, [0, 0, 0, 0]):
                Gi_incomplete = Ai_a_0
            else:
                Gi_incomplete = np.column_stack((Ai_a_0, Ai_a_1))

            null_space = scipy.linalg.null_space(Gi_incomplete.T)

            Gi = np.column_stack((Gi_incomplete, null_space.conjugate()))

            Gi = Gi @ np.real(gates.SWAP)

            G.append(Gi)

    G_last_data = mps[-1].data.copy()
    G_last = G_last_data.T

    if np.allclose(G_last[:, 1], [0, 0]):
        new_vec_1 = scipy.linalg.null_space([G_last[:, 0].T.conjugate()])
        G_last = np.column_stack((G_last[:, 0], new_vec_1))

    G.append(G_last)

    return G


def one_layer_circuit_from_G(G, number_of_qubits):
    """
    Convert G matrices into one old-style layer.
    """
    current_layer_circuit = qiskit.QuantumCircuit(number_of_qubits)

    for i in range(len(G) - 1):
        if G[i].shape == (4, 4):
            current_layer_circuit.unitary(
                G[i],
                [number_of_qubits - 1 - i - 1, number_of_qubits - 1 - i],
                label=f"G{i}",
            )

        elif G[i].shape == (2, 2):
            current_layer_circuit.unitary(
                G[i],
                [number_of_qubits - 1 - i],
                label=f"G{i}",
            )

        else:
            raise ValueError(f"Unsupported G matrix shape: {G[i].shape}")

    current_layer_circuit.unitary(G[-1], [0], label=f"G{len(G)-1}")

    return current_layer_circuit


def repeated_bond2_layers_from_statevector(
    psi,
    fidelity_threshold=0.999,
    max_layers=50,
    normalize=True,
    decompose_gates=True,
):
    """
    Repeated old-style bond-2 layered initializer.

    This repeats:
        current state -> bond-2 MPS -> G layer -> apply G† to residual
    """
    target = np.asarray(psi, dtype=complex)

    if normalize:
        target = normalize_statevector(target)

    number_of_qubits = int(np.log2(len(target)))

    if len(target) != 2**number_of_qubits:
        raise ValueError("The state vector must have size 2^n.")

    current_psi = np.copy(target)
    current_psi = current_psi / np.linalg.norm(current_psi)

    zero_state = np.zeros(len(target), dtype=np.complex128)
    zero_state[0] = 1.0

    layers = []
    history = []

    did_hit_target = False

    for layer_index in range(max_layers):
        current_fidelity_to_zero = float(abs(current_psi[0]) ** 2)
        residual_error = float(np.linalg.norm(current_psi - zero_state))

        history.append(
            {
                "layer": layer_index,
                "fidelity_to_zero": current_fidelity_to_zero,
                "residual_error": residual_error,
            }
        )

        if current_fidelity_to_zero >= fidelity_threshold:
            did_hit_target = True
            break

        mps = bond2_mps_approximation(current_psi)

        G = G_matrices(mps)

        current_layer_circuit = one_layer_circuit_from_G(
            G,
            number_of_qubits,
        )

        unitary = qiskit.quantum_info.Operator.from_circuit(
            current_layer_circuit
        ).data

        current_psi = unitary.conjugate().T @ current_psi
        current_psi = current_psi / np.linalg.norm(current_psi)

        layers.append(current_layer_circuit)

    # Check after final applied layer
    current_fidelity_to_zero = float(abs(current_psi[0]) ** 2)
    residual_error = float(np.linalg.norm(current_psi - zero_state))

    history.append(
        {
            "layer": len(layers),
            "fidelity_to_zero": current_fidelity_to_zero,
            "residual_error": residual_error,
        }
    )

    if current_fidelity_to_zero >= fidelity_threshold:
        did_hit_target = True

    # Reverse layers for preparation circuit
    layers_reversed = list(reversed(layers))

    circuit = qiskit.QuantumCircuit(number_of_qubits)

    for layer in layers_reversed:
        circuit.compose(layer, inplace=True)

    prepared_state = qiskit.quantum_info.Operator.from_circuit(circuit).data @ zero_state

    final_fidelity = state_fidelity(
        target,
        prepared_state,
        normalize=True,
    )

    gate_report = circuit_gate_report(
        circuit,
        decompose=decompose_gates,
    )

    report = {
        "method": "oldstyle_repeated_bond2_layers",
        "bond_dim": 2,
        "n_qubits": number_of_qubits,
        "fidelity_threshold": fidelity_threshold,
        "max_layers": max_layers,
        "chosen_layers": len(layers),
        "hit_target": did_hit_target,
        "fidelity": final_fidelity,
        "final_fidelity_to_zero": current_fidelity_to_zero,
        "final_residual_error": residual_error,
        "history": history,
        "gate_report": gate_report,
    }

    return {
        "circuit": circuit,
        "report": report,
        "history": history,
        "prepared_statevector": prepared_state,
    }


def repeated_bond2_layers_from_function(
    func,
    n_qubits,
    x_min=0.0,
    x_max=1.0,
    mode="amplitude",
    fidelity_threshold=0.999,
    max_layers=50,
    decompose_gates=True,
):
    """
    Function-input version of repeated_bond2_layers_from_statevector.
    """
    psi, x, values = statevector_from_function(
        func,
        n_qubits=n_qubits,
        x_min=x_min,
        x_max=x_max,
        mode=mode,
    )

    result = repeated_bond2_layers_from_statevector(
        psi,
        fidelity_threshold=fidelity_threshold,
        max_layers=max_layers,
        normalize=False,
        decompose_gates=decompose_gates,
    )

    result["x"] = x
    result["values"] = values
    result["statevector"] = psi

    return result
