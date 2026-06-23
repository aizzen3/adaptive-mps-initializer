import numpy as np

from mps_layered_initializer import (
    repeated_bond2_layers_from_function,
    one_layer_bond3_from_function,
)


def main():
    gaussian = lambda x: np.exp(-((x - 0.5) ** 2) / (2 * 0.08 ** 2))

    best_d2 = None
    best_d2_fidelity = -1.0

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
            "D = 2",
            "| layers =", L,
            "| fidelity =", fidelity,
            "| depth =", report["gate_report"]["depth"],
            "| CX =", gates.get("cx", 0),
        )

        if fidelity > best_d2_fidelity:
            best_d2_fidelity = fidelity
            best_d2 = result

    best_report = best_d2["report"]
    best_gates = best_report["gate_report"]["gate_counts"]

    print("\n" + "=" * 80)
    print("Best D = 2 layered result")
    print("=" * 80)
    print("layers =", best_report["chosen_layers"])
    print("fidelity =", best_report["fidelity"])
    print("depth =", best_report["gate_report"]["depth"])
    print("CX =", best_gates.get("cx", 0))
    print("U =", best_gates.get("u", 0))

    print("\n" + "=" * 80)
    print("D = 3 one-layer result")
    print("=" * 80)

    d3 = one_layer_bond3_from_function(
        gaussian,
        n_qubits=8,
        mode="probability",
    )

    d3_report = d3["report"]
    d3_gates = d3_report["gate_report"]["gate_counts"]

    print("layers = 1")
    print("fidelity =", d3_report["fidelity"])
    print("actual bonds =", d3_report["actual_bond_dimensions"])
    print("bond qubits =", d3_report["bond_qubits"])
    print("total circuit qubits =", d3_report["total_circuit_qubits"])
    print("depth =", d3_report["gate_report"]["depth"])
    print("CX =", d3_gates.get("cx", 0))
    print("U =", d3_gates.get("u", 0))

    print("\n" + "=" * 80)
    print("Conclusion")
    print("=" * 80)

    if best_report["fidelity"] >= 0.999:
        print("D=2 layered reaches the target.")
    else:
        print("D=2 layered does NOT reach the 0.999 target.")

    if d3_report["fidelity"] >= 0.999:
        print("D=3 one-layer reaches the target.")
    else:
        print("D=3 one-layer does NOT reach the 0.999 target.")


if __name__ == "__main__":
    main()
