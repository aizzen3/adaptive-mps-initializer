import numpy as np

from mps_initializer import adaptive_mps_from_function


def run_case(name, func, mode="amplitude"):
    print("=" * 80)
    print(name)
    print("=" * 80)

    result = adaptive_mps_from_function(
        func,
        n_qubits=8,
        x_min=0.0,
        x_max=1.0,
        mode=mode,
        fidelity_threshold=0.999,
        max_search_dim=16,
    )

    report = result["report"]

    print("chosen D =", report["chosen_max_bond_dim"])
    print("bond dimensions =", report["actual_bond_dimensions"])
    print("bond qubits =", report["bond_qubits"])
    print("total circuit qubits =", report["total_circuit_qubits"])
    print("fidelity =", report["circuit_preparation_fidelity"])
    print("bond leakage =", report["bond_leakage"])
    print("gate report =", report["gate_report"])
    print()


def main():
    # 1. Smooth sine signal
    sine_func = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)

    # 2. Positive Gaussian signal
    gaussian_func = lambda x: np.exp(-((x - 0.5) ** 2) / (2 * 0.08 ** 2))

    # 3. Square wave / grating-like function
    square_func = lambda x: np.where(np.sin(2 * np.pi * 4 * x) >= 0, 1.0, 0.2)

    # 4. Top-hat function
    top_hat_func = lambda x: np.where((x > 0.35) & (x < 0.65), 1.0, 0.1)

    run_case("Smooth sine amplitude signal", sine_func, mode="amplitude")
    run_case("Gaussian positive signal", gaussian_func, mode="probability")
    run_case("Square-wave positive signal", square_func, mode="probability")
    run_case("Top-hat positive signal", top_hat_func, mode="probability")


if __name__ == "__main__":
    main()
