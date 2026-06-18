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

    print("===== Dense StatePrep vs MPS Circuit Benchmark =====")
    print()

    for n_qubits in range(3, 9):
        psi, x, values = statevector_from_function(
            func,
            n_qubits=n_qubits,
            x_min=0.0,
            x_max=1.0,
            mode="amplitude",
        )

        # Dense Qiskit circuit
        qc_dense = statevector_to_qiskit_circuit(psi)
        dense_report = circuit_gate_report(qc_dense, decompose=True, reps=10)

        # Exact MPS circuit
        mps = statevector_to_mps(psi)
        qc_mps = mps_to_sequential_qiskit_circuit(mps)

        prepared, leakage = physical_statevector_from_mps_circuit(
            qc_mps,
            n_physical=n_qubits,
        )

        mps_fidelity = state_fidelity(psi, prepared)
        mps_report_raw = circuit_gate_report(qc_mps, decompose=False)
        mps_report_dec = circuit_gate_report(qc_mps, decompose=True, reps=10)

        print(f"n_qubits = {n_qubits}")
        print("statevector length =", len(psi))
        print("MPS exact bond dimensions =", mps.bond_dimensions())
        print("MPS exact max D =", mps.max_bond_dimension())
        print("MPS circuit total qubits =", qc_mps.num_qubits)
        print("MPS bond leakage =", leakage)
        print("MPS preparation fidelity =", mps_fidelity)

        print("\nDense decomposed:")
        print("depth =", dense_report["depth"])
        print("size =", dense_report["size"])
        print("gates =", dense_report["gate_counts"])

        print("\nMPS raw:")
        print("depth =", mps_report_raw["depth"])
        print("size =", mps_report_raw["size"])
        print("gates =", mps_report_raw["gate_counts"])

        print("\nMPS decomposed:")
        print("depth =", mps_report_dec["depth"])
        print("size =", mps_report_dec["size"])
        print("gates =", mps_report_dec["gate_counts"])

        print("-" * 70)


if __name__ == "__main__":
    main()
