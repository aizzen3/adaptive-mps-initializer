import numpy as np

from mps_initializer import (
    statevector_from_function,
    statevector_to_mps,
    statevector_to_qiskit_circuit,
    mps_to_sequential_qiskit_circuit,
    physical_statevector_from_mps_circuit,
    state_fidelity,
    circuit_gate_report,
)


def main():
    func = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)

    print("===== Dense vs Exact MPS vs Compressed MPS =====")
    print()

    for n_qubits in range(5, 9):
        psi, x, values = statevector_from_function(
            func,
            n_qubits=n_qubits,
            x_min=0.0,
            x_max=1.0,
            mode="amplitude",
        )

        # Dense Qiskit StatePreparation
        qc_dense = statevector_to_qiskit_circuit(psi)
        dense = circuit_gate_report(qc_dense, decompose=True, reps=10)

        # Exact MPS
        mps_exact = statevector_to_mps(psi)
        qc_exact = mps_to_sequential_qiskit_circuit(mps_exact)

        prepared_exact, leakage_exact = physical_statevector_from_mps_circuit(
            qc_exact,
            n_physical=n_qubits,
        )

        exact_fid = state_fidelity(psi, prepared_exact)
        exact_report = circuit_gate_report(qc_exact, decompose=True, reps=10)

        # Compressed MPS, force D=3
        mps_comp = statevector_to_mps(psi, max_bond_dim=3)
        qc_comp = mps_to_sequential_qiskit_circuit(mps_comp)

        prepared_comp, leakage_comp = physical_statevector_from_mps_circuit(
            qc_comp,
            n_physical=n_qubits,
        )

        comp_fid = state_fidelity(psi, prepared_comp)
        comp_report = circuit_gate_report(qc_comp, decompose=True, reps=10)

        print(f"n_qubits = {n_qubits}")
        print("statevector length =", len(psi))

        print("\nDense StatePreparation")
        print("depth =", dense["depth"])
        print("size =", dense["size"])
        print("gates =", dense["gate_counts"])

        print("\nExact MPS")
        print("bond dimensions =", mps_exact.bond_dimensions())
        print("max D =", mps_exact.max_bond_dimension())
        print("total qubits =", qc_exact.num_qubits)
        print("leakage =", leakage_exact)
        print("fidelity =", exact_fid)
        print("depth =", exact_report["depth"])
        print("size =", exact_report["size"])
        print("gates =", exact_report["gate_counts"])

        print("\nCompressed MPS, max_bond_dim=3")
        print("bond dimensions =", mps_comp.bond_dimensions())
        print("max D =", mps_comp.max_bond_dimension())
        print("total qubits =", qc_comp.num_qubits)
        print("leakage =", leakage_comp)
        print("fidelity =", comp_fid)
        print("depth =", comp_report["depth"])
        print("size =", comp_report["size"])
        print("gates =", comp_report["gate_counts"])

        print("-" * 70)


if __name__ == "__main__":
    main()
