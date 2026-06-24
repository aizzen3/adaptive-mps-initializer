import numpy as np

from mps_layered_initializer import simulate_bond2, simulate_bond3


gaussian = lambda x: np.exp(
    -((x - 0.5) ** 2) / (2 * 0.08**2)
)

d2 = simulate_bond2(
    gaussian,
    n_qubits=8,
    mode="probability",
    max_layers=20,
)

d3 = simulate_bond3(
    gaussian,
    n_qubits=8,
    mode="probability",
    max_layers=7,
)

print("Bond-2 best fidelity:", d2["report"]["fidelity"])
print("Bond-2 best layers:", d2["report"]["chosen_layers"])

print("Bond-3 best fidelity:", d3["report"]["fidelity"])
print("Bond-3 best layers:", d3["report"]["chosen_layers"])
