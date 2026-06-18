import numpy as np

from mps_initializer import adaptive_mps_from_function


def main():
    # Any arbitrary function can go here.
    func = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)

    result = adaptive_mps_from_function(
        func,
        n_qubits=8,
        x_min=0.0,
        x_max=1.0,
        mode="amplitude",
        fidelity_threshold=0.999,
        max_search_dim=16,
    )

    report = result["report"]
    qc = result["circuit"]

    print("===== Adaptive MPS API Demo =====")
    print("chosen D =", report["chosen_max_bond_dim"])
    print("bond dimensions =", report["actual_bond_dimensions"])
    print("bond qubits =", report["bond_qubits"])
    print("total circuit qubits =", report["total_circuit_qubits"])
    print("fidelity =", report["circuit_preparation_fidelity"])
    print("bond leakage =", report["bond_leakage"])
    print("gate report =", report["gate_report"])

    print("\nCircuit:")
    print(qc)


if __name__ == "__main__":
    main()
