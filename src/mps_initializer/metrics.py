import numpy as np


def normalize_statevector(psi):
    """
    Normalize a quantum statevector.
    """
    psi = np.asarray(psi, dtype=complex)

    norm = np.linalg.norm(psi)

    if norm == 0:
        raise ValueError("Cannot normalize zero vector.")

    return psi / norm


def state_fidelity(psi, phi, normalize=True):
    """
    Compute fidelity between two statevectors.

    Fidelity = |<psi|phi>|^2

    If both states are identical, fidelity = 1.
    """

    psi = np.asarray(psi, dtype=complex)
    phi = np.asarray(phi, dtype=complex)

    if psi.shape != phi.shape:
        raise ValueError("Statevectors must have the same shape.")

    if normalize:
        psi = normalize_statevector(psi)
        phi = normalize_statevector(phi)

    overlap = np.vdot(psi, phi)

    return float(np.abs(overlap) ** 2)
