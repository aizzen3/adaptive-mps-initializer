import numpy as np

from .state import QuantumState


def singular_values_across_cuts(data, normalize=True):
    """
    Compute singular values across every bipartition/cut.

    For n qubits, there are n-1 cuts:
        cut 1: qubit 1 | rest
        cut 2: qubits 1-2 | rest
        ...
    """

    state = QuantumState(data, normalize=normalize)
    psi = state.data
    n_qubits = state.n_qubits

    all_singular_values = []

    for cut in range(1, n_qubits):
        left_dim = 2 ** cut
        right_dim = 2 ** (n_qubits - cut)

        matrix = psi.reshape(left_dim, right_dim)

        s = np.linalg.svd(matrix, compute_uv=False)

        all_singular_values.append(s)

    return all_singular_values


def entanglement_entropy_from_singular_values(s, cutoff=1e-15):
    """
    Entanglement entropy:

        S = - sum p_i log2(p_i)

    where p_i = singular_value_i^2.
    """

    p = np.abs(s) ** 2
    p = p[p > cutoff]

    if len(p) == 0:
        return 0.0

    return float(-np.sum(p * np.log2(p)))


def entanglement_entropy_profile(data, normalize=True):
    """
    Compute entanglement entropy across all cuts.
    """

    singular_values = singular_values_across_cuts(data, normalize=normalize)

    return np.array([
        entanglement_entropy_from_singular_values(s)
        for s in singular_values
    ])


def rank_profile(data, cutoff=1e-12, normalize=True):
    """
    Compute numerical Schmidt rank across all cuts.
    """

    singular_values = singular_values_across_cuts(data, normalize=normalize)

    return np.array([
        int(np.sum(s > cutoff))
        for s in singular_values
    ])


def cumulative_rank_profile(data, weight=0.999999, normalize=True):
    """
    Compute how many singular values are needed to capture given weight.

    Example:
        weight = 0.999999 means 99.9999 percent of probability weight.
    """

    singular_values = singular_values_across_cuts(data, normalize=normalize)

    ranks = []

    for s in singular_values:
        p = np.abs(s) ** 2
        total = np.sum(p)

        if total == 0:
            ranks.append(0)
            continue

        p = p / total
        cumulative = np.cumsum(p)

        r = int(np.searchsorted(cumulative, weight) + 1)
        ranks.append(r)

    return np.array(ranks)
