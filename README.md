# Adaptive MPS Initializer

This package prepares quantum states from 1D functions or statevectors using adaptive Matrix Product State compression.

It builds Qiskit circuits and automatically chooses the smallest bond dimension needed for a target fidelity.

## Installation

```bash
pip install git+https://github.com/aizzen3/adaptive-mps-initializer.git
```

## Usage

```python
import numpy as np
from mps_initializer import prepare_from_function

func = lambda x: np.sin(2 * np.pi * x)

result = prepare_from_function(
    func,
    n_qubits=8,
    mode="amplitude",
    fidelity_threshold=0.999,
)

circuit = result["circuit"]
report = result["report"]

print(report["chosen_max_bond_dim"])
print(report["circuit_preparation_fidelity"])

circuit.draw("mpl")
```

## Statevector input

```python
from mps_initializer import prepare_from_statevector

result = prepare_from_statevector(psi, fidelity_threshold=0.999)
```

## Status

Research prototype. API may change.
