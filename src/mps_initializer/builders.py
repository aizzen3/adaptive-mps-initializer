import numpy as np

from .state import QuantumState


def statevector_from_amplitudes(values, normalize=True):
    """
    Convert arbitrary amplitude values into a normalized quantum statevector.

    Use this when your function values ARE the amplitudes.

    Example:
        f(x) = sin(x)
        psi_j = f(x_j)
    """

    psi = np.asarray(values, dtype=complex)

    if psi.ndim != 1:
        raise ValueError("Input values must be a 1D array.")

    state = QuantumState(psi, normalize=normalize)

    return state.data


def statevector_from_probabilities(values, normalize=True):
    """
    Convert non-negative function values into quantum amplitudes.

    Use this when your function represents a positive signal/probability weight.

    Formula:
        psi_j = sqrt(f_j / sum(f_j))
    """

    f = np.asarray(values, dtype=float)

    if f.ndim != 1:
        raise ValueError("Input values must be a 1D array.")

    if np.any(f < 0):
        raise ValueError("Probability/signal values must be non-negative.")

    total = np.sum(f)

    if total == 0:
        raise ValueError("Signal sum cannot be zero.")

    psi = np.sqrt(f / total).astype(complex)

    if normalize:
        psi = psi / np.linalg.norm(psi)

    return psi


def sample_grid(n_qubits, x_min=0.0, x_max=1.0, endpoint=False):
    """
    Create a 1D grid with 2**n_qubits points.
    """

    n_points = 2 ** n_qubits

    return np.linspace(x_min, x_max, n_points, endpoint=endpoint)


def statevector_from_function(
    func,
    n_qubits,
    x_min=0.0,
    x_max=1.0,
    mode="amplitude",
    endpoint=False,
):
    """
    Sample an arbitrary function and convert it into a quantum statevector.

    Parameters
    ----------
    func:
        Python function, for example:
            lambda x: np.sin(2*np.pi*x)

    n_qubits:
        Number of qubits. Number of samples = 2**n_qubits.

    mode:
        "amplitude"   -> use function values directly as amplitudes
        "probability" -> use sqrt(f/sum(f)), requiring f >= 0
    """

    x = sample_grid(
        n_qubits=n_qubits,
        x_min=x_min,
        x_max=x_max,
        endpoint=endpoint,
    )

    values = func(x)

    if mode == "amplitude":
        psi = statevector_from_amplitudes(values)

    elif mode == "probability":
        psi = statevector_from_probabilities(values)

    else:
        raise ValueError("mode must be either 'amplitude' or 'probability'.")

    return psi, x, values
