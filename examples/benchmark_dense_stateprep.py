import numpy as np

from mps_initializer import (
    statevector_from_function,
    analyze_statevector,
    statevector_to_qiskit_circuit,
    circuit_gate_report,
)


def main():
    func = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)

    print("===== Dense StatePreparation Benchmark =====")
    print()

    for n_qubits in range(3, 9):
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
            max_search_dim=32,
        )

        qc = statevector_to_qiskit_circuit(psi)

        raw = circuit_gate_report(qc, decompose=False)
        dec = circuit_gate_report(qc, decompose=True, reps=10)

        best = report["best_compressed_result"]

        print(f"n_qubits = {n_qubits}")
        print("statevector length =", len(psi))
        print("exact max bond dimension =", report["exact_max_bond_dimension"])

        if best is not None:
            print("best compressed D =", best["actual_max_bond_dim"])
            print("compressed fidelity =", best["fidelity"])
        else:
            print("best compressed D = not found")

        print("dense raw depth =", raw["depth"])
        print("dense decomposed depth =", dec["depth"])
        print("dense decomposed size =", dec["size"])
        print("dense decomposed gates =", dec["gate_counts"])
        print("-" * 60)


if __name__ == "__main__":
    main()
