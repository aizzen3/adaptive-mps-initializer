import numpy as np

from mps_initializer import (
    statevector_from_function,
    analyze_statevector,
)


def main():
    # You can replace this with any function later
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

    print("===== MPS Analysis Report =====")
    print("n_qubits =", report["n_qubits"])
    print("statevector length =", report["statevector_length"])
    print("norm =", report["norm"])

    print("\nExact MPS")
    print("bond dimensions =", report["exact_bond_dimensions"])
    print("max bond dimension =", report["exact_max_bond_dimension"])
    print("fidelity =", report["exact_fidelity"])

    print("\nDiagnostics")
    print("entropy profile =", report["entropy_profile"])
    print("rank profile =", report["rank_profile"])
    print("cumulative rank profile =", report["cumulative_rank_profile"])

    print("\nBest compressed result")
    print(report["best_compressed_result"])


if __name__ == "__main__":
    main()
