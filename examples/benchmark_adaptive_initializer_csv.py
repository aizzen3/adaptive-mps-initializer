import csv
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
    output_file = "adaptive_benchmark.csv"

    rows = []

    for n_qubits in range(3, 11):
        psi, x, values = statevector_from_function(
            func,
            n_qubits=n_qubits,
            x_min=0.0,
            x_max=1.0,
            mode="amplitude",
        )

        qc_mps, mps, report = adaptive_mps_initializer(
            psi,
            fidelity_threshold=fidelity_threshold,
            max_search_dim=32,
        )

        mps_gate_report = report["gate_report"]
        mps_gates = mps_gate_report["gate_counts"]

        qc_dense = statevector_to_qiskit_circuit(psi)
        dense_report = circuit_gate_report(
            qc_dense,
            decompose=True,
            reps=10,
        )

        dense_gates = dense_report["gate_counts"]

        row = {
            "n_qubits": n_qubits,
            "statevector_length": len(psi),
            "chosen_D": report["chosen_max_bond_dim"],
            "bond_qubits": report["bond_qubits"],
            "total_mps_circuit_qubits": report["total_circuit_qubits"],
            "mps_fidelity": report["circuit_preparation_fidelity"],
            "mps_cx": mps_gates.get("cx", 0),
            "mps_u": mps_gates.get("u", 0),
            "mps_depth": mps_gate_report["depth"],
            "dense_cx": dense_gates.get("cx", 0),
            "dense_u": dense_gates.get("u", 0),
            "dense_depth": dense_report["depth"],
        }

        rows.append(row)

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", output_file)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
