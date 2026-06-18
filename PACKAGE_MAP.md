# MPS Initializer Package Map

## Core idea

This package takes an arbitrary function or statevector and analyzes how well it can be represented as an MPS.

Workflow:

function values
    -> statevector
    -> MPS tensors
    -> bond dimensions / entropy / rank
    -> compressed MPS fidelity

## Files

src/mps_initializer/state.py
    Defines QuantumState.
    Checks statevector length, normalizes it, finds number of qubits.

src/mps_initializer/mps.py
    Converts dense statevector to MPS using SVD.
    Converts MPS back to statevector.

src/mps_initializer/metrics.py
    Computes fidelity and normalizes statevectors.

src/mps_initializer/diagnostics.py
    Computes singular values, entanglement entropy, rank profile,
    cumulative rank profile.

src/mps_initializer/builders.py
    Converts arbitrary sampled functions into statevectors.

src/mps_initializer/compression.py
    Sweeps bond dimension D and checks fidelity.

src/mps_initializer/report.py
    High-level analyzer.
    Main function: analyze_statevector(psi)

src/mps_initializer/__init__.py
    Makes imports easy.
