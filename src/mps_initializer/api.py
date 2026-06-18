"""
Public API for the adaptive MPS initializer.

This file contains easy-to-use functions that hide the internal package steps.

Main user functions:
    adaptive_mps_from_statevector(...)
    adaptive_mps_from_function(...)
"""

from .builders import statevector_from_function
from .adaptive import adaptive_mps_initializer


def adaptive_mps_from_statevector(
    psi,
    fidelity_threshold=0.999,
    max_search_dim=64,
    cutoff=1e-12,
    normalize=True,
):
    """
    Build an adaptive MPS initializer directly from a statevector.

    Parameters
    ----------
    psi:
        Input quantum statevector. Length must be 2**n.

    fidelity_threshold:
        Minimum target fidelity.

    max_search_dim:
        Maximum bond dimension to search.

    cutoff:
        Singular-value cutoff.

    normalize:
        If True, normalize the input statevector.

    Returns
    -------
    result:
        Dictionary containing:
            circuit
            mps
            report
    """

    qc, mps, report = adaptive_mps_initializer(
        psi,
        fidelity_threshold=fidelity_threshold,
        max_search_dim=max_search_dim,
        cutoff=cutoff,
        normalize=normalize,
    )

    return {
        "circuit": qc,
        "mps": mps,
        "report": report,
    }


def adaptive_mps_from_function(
    func,
    n_qubits,
    x_min=0.0,
    x_max=1.0,
    mode="amplitude",
    fidelity_threshold=0.999,
    max_search_dim=64,
    cutoff=1e-12,
):
    """
    Build an adaptive MPS initializer from an arbitrary 1D function.

    Workflow:
        function
            -> sampled values
            -> statevector
            -> compressed MPS
            -> Qiskit circuit

    Parameters
    ----------
    func:
        Python function. Example:
            lambda x: np.sin(2*np.pi*x)

    n_qubits:
        Number of qubits. The function is sampled at 2**n_qubits points.

    x_min, x_max:
        Sampling interval.

    mode:
        "amplitude":
            use function values directly as amplitudes, then normalize.

        "probability":
            require non-negative function values and use sqrt(f/sum(f)).

    fidelity_threshold:
        Minimum target fidelity.

    max_search_dim:
        Maximum bond dimension to search.

    cutoff:
        Singular-value cutoff.

    Returns
    -------
    result:
        Dictionary containing:
            x
            values
            statevector
            circuit
            mps
            report
    """

    psi, x, values = statevector_from_function(
        func,
        n_qubits=n_qubits,
        x_min=x_min,
        x_max=x_max,
        mode=mode,
    )

    qc, mps, report = adaptive_mps_initializer(
        psi,
        fidelity_threshold=fidelity_threshold,
        max_search_dim=max_search_dim,
        cutoff=cutoff,
        normalize=False,
    )

    return {
        "x": x,
        "values": values,
        "statevector": psi,
        "circuit": qc,
        "mps": mps,
        "report": report,
    }
