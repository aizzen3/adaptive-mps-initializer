"""
Adaptive MPS Initializer

User-friendly API for preparing quantum states from functions or statevectors
using adaptive matrix product state compression.
"""

from .api import (
    adaptive_mps_from_function,
    adaptive_mps_from_statevector,
)

from .circuits import physical_statevector_from_mps_circuit
from .metrics import state_fidelity
from .report import analyze_statevector
from .state import QuantumState


# Short friendly aliases
prepare_from_function = adaptive_mps_from_function
prepare_from_statevector = adaptive_mps_from_statevector


__all__ = [
    "adaptive_mps_from_function",
    "adaptive_mps_from_statevector",
    "prepare_from_function",
    "prepare_from_statevector",
    "physical_statevector_from_mps_circuit",
    "state_fidelity",
    "analyze_statevector",
    "QuantumState",
]
