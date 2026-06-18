import numpy as np

from mps_initializer import (
    statevector_from_function,
    adaptive_mps_initializer,
)


def main():
    n_qubits = 8
    fidelity_threshold = 0.999

    # Replace this function with any signal/function later
    func = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)

    psi, x, values = statevector_from_function(
        func,
        n_qubits=n_qubits,
        x_min=0.0,
        x_max=1.0,
        mode="amplitude",
    )

    qc, mps, report = adaptive_mps_initializer(
        psi,
        fidelity_threshold=fidelity_threshold,
        max_search_dim=16,
    )

    print("===== Adaptive MPS Initializer Demo =====")
    print("n_qubits =", report["n_qubits"])
    print("statevector length =", report["statevector_length"])
    print("target fidelity threshold =", report["target_fidelity_threshold"])

    print("\nChosen compressed MPS")
    print("chosen D =", report["chosen_max_bond_dim"])
    print("bond dimensions =", report["actual_bond_dimensions"])
    print("bond qubits =", report["bond_qubits"])

    print("\nCircuit")
    print("total circuit qubits =", report["total_circuit_qubits"])
    print("circuit preparation fidelity =", report["circuit_preparation_fidelity"])
    print("bond leakage =", report["bond_leakage"])
    print("gate report =", report["gate_report"])

    print("\nCircuit object:")
    print(qc)


if __name__ == "__main__":
    main()
