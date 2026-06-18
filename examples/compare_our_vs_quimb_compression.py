import numpy as np

from mps_initializer import (
    statevector_from_function,
    statevector_to_mps,
    compression_sweep,
    quimb_compression_sweep,
)


def main():
    func = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)

    n_qubits = 8

    psi, x, values = statevector_from_function(
        func,
        n_qubits=n_qubits,
        x_min=0.0,
        x_max=1.0,
        mode="amplitude",
    )

    mps = statevector_to_mps(psi)

    dims = [1, 2, 3, 4]

    our_results = compression_sweep(
        psi,
        max_bond_dims=dims,
    )

    quimb_results = quimb_compression_sweep(
        mps,
        target_statevector=psi,
        max_bond_dims=dims,
    )

    print("===== Our SVD Compression vs Quimb Compression =====")
    print("n_qubits =", n_qubits)
    print("statevector length =", len(psi))
    print("exact MPS bonds =", mps.bond_dimensions())
    print()

    print("D | our fidelity | our bonds | quimb fidelity | quimb bonds")
    print("-" * 80)

    for ours, qb in zip(our_results, quimb_results):
        print(
            ours["max_bond_dim_allowed"],
            "|",
            ours["fidelity"],
            "|",
            ours["bond_dimensions"],
            "|",
            qb["fidelity"],
            "|",
            qb["quimb_bond_sizes"],
        )


if __name__ == "__main__":
    main()
