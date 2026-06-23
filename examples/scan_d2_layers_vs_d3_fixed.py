import numpy as np

from mps_layered_initializer import (
    repeated_bond2_layers_from_function,
    fixed_bond_mps_layer_from_function,
)


def main():
    gaussian = lambda x: np.exp(-((x - 0.5) ** 2) / (2 * 0.08 ** 2))

    best_d2 = None
    best_fidelity = -1.0

    print("=" * 80)
    print("Scanning D = 2 repeated layers")
    print("=" * 80)

    for L in range(1, 21):
        result = repeated_bond2_layers_from_function(
            gaussian,
            n_qubits=8,
            mode="probability",
            fidelity_threshold=0.999,
            max_layers=L,
        )

        report = result["report"]
        gates = report["gate_report"]["gate_counts"]

        fidelity = report["fidelity"]

        print(
            "layers =", L,
            "| fidelity =", fidelity,
            "| depth =", report["gate_report"]["depth"],
            "| CX =", gates.get("cx", 0),
        )

        if fidelity > best_fidelity:
            best_fidelity = fidelity
            best_d2 = result

    best_report = best_d2["report"]
    best_gates = best_report["gate_report"]["gate_counts"]

    print("\n" + "=" * 80)
    print("Best D = 2 layered result")
    print("=" * 80)
    print("best layers =", best_report["chosen_layers"])
    print("best fidelity =", best_report["fidelity"])
    print("depth =", best_report["gate_report"]["depth"])
    print("CX gates =", best_gates.get("cx", 0))
    print("U gates =", best_gates.get("u", 0))

    print("\n" + "=" * 80)
    print("D = 3 one-shot fixed-bond result")
    print("=" * 80)

    d3 = fixed_bond_mps_layer_from_function(
        gaussian,
        n_qubits=8,
        mode="probability",
        bond_dim=3,
    )

    d3_report = d3["report"]
    d3_gates = d3_report["gate_report"]["gate_counts"]

    print("fidelity =", d3_report["fidelity"])
    print("actual bonds =", d3_report["actual_bond_dimensions"])
    print("bond qubits =", d3_report["bond_qubits"])
    print("total circuit qubits =", d3_report["total_circuit_qubits"])
    print("depth =", d3_report["gate_report"]["depth"])
    print("CX gates =", d3_gates.get("cx", 0))
    print("U gates =", d3_gates.get("u", 0))


if __name__ == "__main__":
    main()
