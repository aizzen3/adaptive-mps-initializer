"""
Clean user-facing API for layered MPS simulations.
"""

from .bond2_layered import repeated_bond2_layers_from_function
from .bond3_repeated_layered import repeated_bond3_layers_from_function


def simulate_bond2(
    func,
    n_qubits,
    x_min=0.0,
    x_max=1.0,
    mode="amplitude",
    max_layers=20,
    fidelity_threshold=0.999,
):
    """
    Simulate the bond-2 repeated layered method.

    This scans layers from 1 to max_layers and returns the best result.
    """
    best_result = None
    best_fidelity = -1.0
    scan_history = []

    for layers in range(1, max_layers + 1):
        result = repeated_bond2_layers_from_function(
            func,
            n_qubits=n_qubits,
            x_min=x_min,
            x_max=x_max,
            mode=mode,
            fidelity_threshold=fidelity_threshold,
            max_layers=layers,
        )

        report = result["report"]
        gates = report["gate_report"]["gate_counts"]

        row = {
            "bond_dim": 2,
            "layers": layers,
            "fidelity": report["fidelity"],
            "depth": report["gate_report"]["depth"],
            "cx": gates.get("cx", 0),
            "u": gates.get("u", 0),
        }

        scan_history.append(row)

        if report["fidelity"] > best_fidelity:
            best_fidelity = report["fidelity"]
            best_result = result

    best_result["report"]["scan_history"] = scan_history
    best_result["report"]["best_layers_from_scan"] = best_result["report"]["chosen_layers"]

    return best_result


def simulate_bond3(
    func,
    n_qubits,
    x_min=0.0,
    x_max=1.0,
    mode="amplitude",
    max_layers=1,
    fidelity_threshold=0.999,
):
    """
    Simulate the bond-3 repeated layered method.

    For max_layers=1, this is the practical one-layer D=3 result.
    For max_layers>1, it scans projected repeated D=3 layers.
    """
    best_result = None
    best_fidelity = -1.0
    scan_history = []

    for layers in range(1, max_layers + 1):
        result = repeated_bond3_layers_from_function(
            func,
            n_qubits=n_qubits,
            x_min=x_min,
            x_max=x_max,
            mode=mode,
            fidelity_threshold=1.1,  # force scan; do not stop early
            max_layers=layers,
        )

        report = result["report"]
        gates = report["gate_report"]["gate_counts"]

        row = {
            "bond_dim": 3,
            "layers": report["chosen_layers"],
            "requested_layers": layers,
            "fidelity": report["fidelity"],
            "depth": report["gate_report"]["depth"],
            "cx": gates.get("cx", 0),
            "u": gates.get("u", 0),
            "bond_leakage": report["bond_leakage"],
        }

        scan_history.append(row)

        if report["fidelity"] > best_fidelity:
            best_fidelity = report["fidelity"]
            best_result = result

    best_result["report"]["scan_history"] = scan_history
    best_result["report"]["best_layers_from_scan"] = best_result["report"]["chosen_layers"]

    return best_result
