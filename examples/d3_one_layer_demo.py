import numpy as np

from mps_layered_initializer import one_layer_bond3_from_function


def main():
    gaussian = lambda x: np.exp(-((x - 0.5) ** 2) / (2 * 0.08 ** 2))

    result = one_layer_bond3_from_function(
        gaussian,
        n_qubits=8,
        mode="probability",
    )

    report = result["report"]
    gates = report["gate_report"]["gate_counts"]

    print("=" * 80)
    print("D = 3 one-layer MPS preparation")
    print("=" * 80)
    print("method =", report["method"])
    print("physical qubits =", report["n_physical_qubits"])
    print("bond dimension =", report["bond_dim"])
    print("actual bonds =", report["actual_bond_dimensions"])
    print("actual max bond =", report["actual_max_bond_dim"])
    print("bond qubits =", report["bond_qubits"])
    print("total circuit qubits =", report["total_circuit_qubits"])
    print("fidelity =", report["fidelity"])
    print("bond leakage =", report["bond_leakage"])
    print("depth =", report["gate_report"]["depth"])
    print("CX gates =", gates.get("cx", 0))
    print("U gates =", gates.get("u", 0))


if __name__ == "__main__":
    main()
