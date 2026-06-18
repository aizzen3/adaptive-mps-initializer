import numpy as np

from mps_initializer import (
    statevector_from_function,
    adaptive_mps_initializer,
    statevector_to_qiskit_circuit,
    circuit_gate_report,
)


def main():
    func = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)

    fidelity_threshold = 0.999

    print("===== Adaptive MPS Initializer Scaling Benchmark =====")
    print("target fidelity threshold =", fidelity_threshold)
    print()

    header = (
        "n | len(psi) | chosen D | bond qubits | total qubits | "
        "MPS fidelity | MPS CX | MPS depth | Dense CX | Dense depth"
    )

    print(header)
    print("-" * len(header))

    for n_qubits in range(3, 11):
        psi, x, values = statevector_from_function(
            func,
            n_qubits=n_qubits,
            x_min=0.0,
            x_max=1.0,
            mode="amplitude",
        )

        # Adaptive MPS circuit
        qc_mps, mps, report = adaptive_mps_initializer(
            psi,
            fidelity_threshold=fidelity_threshold,
            max_search_dim=32,
        )

        mps_gate_report = report["gate_report"]
        mps_gates = mps_gate_report["gate_counts"]

        mps_cx = mps_gates.get("cx", 0)
        mps_depth = mps_gate_report["depth"]

        # Dense Qiskit StatePreparation baseline
        qc_dense = statevector_to_qiskit_circuit(psi)
        dense_report = circuit_gate_report(
            qc_dense,
            decompose=True,
            reps=10,
        )

        dense_gates = dense_report["gate_counts"]
        dense_cx = dense_gates.get("cx", 0)
        dense_depth = dense_report["depth"]

        print(
            n_qubits,
            "|",
            len(psi),
            "|",
            report["chosen_max_bond_dim"],
            "|",
            report["bond_qubits"],
            "|",
            report["total_circuit_qubits"],
            "|",
            round(report["circuit_preparation_fidelity"], 12),
            "|",
            mps_cx,
            "|",
            mps_depth,
            "|",
            dense_cx,
            "|",
            dense_depth,
        )


if __name__ == "__main__":
    main()
