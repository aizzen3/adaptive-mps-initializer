import numpy as np

from mps_initializer import (
    QuantumState,
    statevector_to_mps,
    state_fidelity,
    statevector_from_function,
    analyze_statevector,
)


def test_quantum_state_normalization():
    psi = [1, 1, 1, 1]
    state = QuantumState(psi)

    assert state.n_qubits == 2
    assert np.isclose(state.norm(), 1.0)


def test_product_state_bond_dimension():
    psi = np.array([1, 1, 1, 1], dtype=complex)

    mps = statevector_to_mps(psi)

    assert mps.bond_dimensions() == [1]
    assert mps.max_bond_dimension() == 1


def test_bell_state_bond_dimension():
    psi = np.array([1, 0, 0, 1], dtype=complex)
    psi = psi / np.linalg.norm(psi)

    mps = statevector_to_mps(psi)
    reconstructed = mps.to_statevector()

    assert mps.bond_dimensions() == [2]
    assert mps.max_bond_dimension() == 2
    assert np.isclose(state_fidelity(psi, reconstructed), 1.0)


def test_arbitrary_function_analysis():
    func = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)

    psi, x, values = statevector_from_function(
        func,
        n_qubits=6,
        x_min=0.0,
        x_max=1.0,
        mode="amplitude",
    )

    report = analyze_statevector(
        psi,
        fidelity_threshold=0.999,
        max_search_dim=16,
    )

    assert report["n_qubits"] == 6
    assert report["statevector_length"] == 64
    assert np.isclose(report["norm"], 1.0)
    assert report["exact_max_bond_dimension"] == 4
    assert report["best_compressed_result"]["actual_max_bond_dim"] == 3
