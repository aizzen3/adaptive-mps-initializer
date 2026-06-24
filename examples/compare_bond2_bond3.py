import numpy as np

from mps_layered_initializer import (
    simulate_bond2,
    simulate_bond3,
)


def main():
    gaussian = lambda x: np.exp(
        -((x - 0.5) ** 2) / (2 * 0.08**2)
    )

    print("=" * 80)
    print("Bond-2 layered simulation")
    print("=" * 80)

    d2 = simulate_bond2(
        gaussian,
        n_qubits=8,
        mode="probability",
        max_layers=20,
        fidelity_threshold=0.999,
    )

    d2_report = d2["report"]
    d2_gates = d2_report["gate_report"]["gate_counts"]

    print("best layers =", d2_report["chosen_layers"])
    print("fidelity =", d2_report["fidelity"])
    print("depth =", d2_report["gate_report"]["depth"])
    print("CX =", d2_gates.get("cx", 0))
    print("U =", d2_gates.get("u", 0))

    print()
    print("=" * 80)
    print("Bond-3 layered simulation")
    print("=" * 80)

    d3 = simulate_bond3(
        gaussian,
        n_qubits=8,
        mode="probability",
        max_layers=7,
        fidelity_threshold=0.999,
    )

    d3_report = d3["report"]
    d3_gates = d3_report["gate_report"]["gate_counts"]

    print("best layers =", d3_report["chosen_layers"])
    print("fidelity =", d3_report["fidelity"])
    print("bond leakage =", d3_report["bond_leakage"])
    print("depth =", d3_report["gate_report"]["depth"])
    print("CX =", d3_gates.get("cx", 0))
    print("U =", d3_gates.get("u", 0))

    print()
    print("=" * 80)
    print("Short conclusion")
    print("=" * 80)

    if d2_report["fidelity"] >= 0.999:
        print("Bond-2 reaches the 0.999 target.")
    else:
        print("Bond-2 does NOT reach the 0.999 target.")

    if d3_report["fidelity"] >= 0.999:
        print("Bond-3 reaches the 0.999 target.")
    else:
        print("Bond-3 does NOT reach the 0.999 target.")


if __name__ == "__main__":
    main()
