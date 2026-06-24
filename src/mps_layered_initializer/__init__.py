"""
Layered MPS initializer.

Clean user-facing imports:

    from mps_layered_initializer import simulate_bond2, simulate_bond3
"""

from .api import simulate_bond2, simulate_bond3

from .bond2_layered import (
    repeated_bond2_layers_from_statevector,
    repeated_bond2_layers_from_function,
)

from .bond3_repeated_layered import (
    repeated_bond3_layers_from_statevector,
    repeated_bond3_layers_from_function,
)

from .bond3_ancilla_layer import (
    bond3_ancilla_layer_from_statevector,
    bond3_ancilla_layer_from_function,
)

__all__ = [
    "simulate_bond2",
    "simulate_bond3",
    "repeated_bond2_layers_from_statevector",
    "repeated_bond2_layers_from_function",
    "repeated_bond3_layers_from_statevector",
    "repeated_bond3_layers_from_function",
    "bond3_ancilla_layer_from_statevector",
    "bond3_ancilla_layer_from_function",
]
