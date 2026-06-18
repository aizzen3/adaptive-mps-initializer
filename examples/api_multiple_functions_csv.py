import csv
import numpy as np

from mps_initializer import adaptive_mps_from_function


def run_case(name, func, mode="amplitude"):
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
    gates = report["gate_report"]["gate_counts"]

    return {
        "name": name,
        "mode": mode,
        "chosen_D": report["chosen_max_bond_dim"],
        "bond_dimensions": str(report["actual_bond_dimensions"]),
        "bond_qubits": report["bond_qubits"],
        "total_circuit_qubits": report["total_circuit_qubits"],
        "fidelity": report["circuit_preparation_fidelity"],
        "bond_leakage": report["bond_leakage"],
        "cx_gates": gates.get("cx", 0),
        "u_gates": gates.get("u", 0),
        "depth": report["gate_report"]["depth"],
        "size": report["gate_report"]["size"],
    }


def main():
    sine_func = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(6 * np.pi * x)
    gaussian_func = lambda x: np.exp(-((x - 0.5) ** 2) / (2 * 0.08 ** 2))
    square_func = lambda x: np.where(np.sin(2 * np.pi * 4 * x) >= 0, 1.0, 0.2)
    top_hat_func = lambda x: np.where((x > 0.35) & (x < 0.65), 1.0, 0.1)

    rows = [
        run_case("smooth_sine", sine_func, mode="amplitude"),
        run_case("gaussian", gaussian_func, mode="probability"),
        run_case("square_wave", square_func, mode="probability"),
        run_case("top_hat", top_hat_func, mode="probability"),
    ]

    output_file = "api_multiple_functions_results.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Saved:", output_file)

    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
