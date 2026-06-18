import numpy as np

from mps_initializer import (
    statevector_from_function,
    analyze_statevector,
    statevector_to_qiskit_circuit,
    circuit_statevector,
    state_fidelity,
)


def main():
    n_qubits = 6

    # Replace this with any function you want
    func = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)

    psi, x, values = statevector_from_function(
        func,
        n_qubits=n_qubits,
        x_min=0.0,
        x_max=1.0,
        mode="amplitude",
    )

    report = analyze_statevector(
        psi,
        fidelity_threshold=0.999,
        max_search_dim=16,
    )

    qc = statevector_to_qiskit_circuit(psi)
    prepared = circuit_statevector(qc)

    print("===== Function to Circuit Example =====")
    print("n_qubits =", n_qubits)
    print("statevector length =", len(psi))

    print("\nMPS analysis")
    print("exact bond dimensions =", report["exact_bond_dimensions"])
    print("exact max bond dimension =", report["exact_max_bond_dimension"])
    print("best compressed result =", report["best_compressed_result"])

    print("\nQiskit circuit")
    print("number of qubits =", qc.num_qubits)
    print("circuit depth =", qc.depth())
    print("state preparation fidelity =", state_fidelity(psi, prepared))

    print("\nCircuit:")
    print(qc)


if __name__ == "__main__":
    main()
