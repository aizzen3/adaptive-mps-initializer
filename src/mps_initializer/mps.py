import numpy as np

from .state import QuantumState


class MatrixProductState:
    """
    Stores an MPS as a list of tensors.

    Each tensor has shape:

        (left_bond, physical_dimension, right_bond)

    For qubits, physical_dimension = 2.
    """

    def __init__(self, tensors):
        self.tensors = tensors
        self.n_qubits = len(tensors)

    def bond_dimensions(self):
        """
        Returns internal bond dimensions.
        """
        if self.n_qubits <= 1:
            return []

        return [tensor.shape[2] for tensor in self.tensors[:-1]]

    def max_bond_dimension(self):
        """
        Returns the largest internal bond dimension.
        """
        bonds = self.bond_dimensions()

        if len(bonds) == 0:
            return 1

        return max(bonds)

    def to_statevector(self):
        """
        Convert MPS back to dense statevector.
        Useful for checking if decomposition worked.
        """
        psi = self.tensors[0]

        for tensor in self.tensors[1:]:
            psi = np.tensordot(psi, tensor, axes=([-1], [0]))

        psi = np.squeeze(psi, axis=(0, -1))
        return psi.reshape(-1)


def statevector_to_mps(data, max_bond_dim=None, cutoff=1e-12, normalize=True):
    """
    Convert a dense quantum statevector into MPS tensors using SVD.

    Parameters
    ----------
    data:
        Dense statevector.

    max_bond_dim:
        Optional maximum allowed bond dimension.

    cutoff:
        Singular values below this are discarded.

    normalize:
        If True, normalize the input statevector first.
    """

    state = QuantumState(data, normalize=normalize)

    n_qubits = state.n_qubits
    psi = state.data

    tensors = []

    left_bond = 1
    remaining = psi.reshape(1, -1)

    for site in range(n_qubits - 1):
        remaining = remaining.reshape(left_bond * 2, -1)

        U, S, Vh = np.linalg.svd(remaining, full_matrices=False)

        keep = np.sum(S > cutoff)

        if max_bond_dim is not None:
            keep = min(keep, max_bond_dim)

        keep = max(1, int(keep))

        U = U[:, :keep]
        S = S[:keep]
        Vh = Vh[:keep, :]

        tensor = U.reshape(left_bond, 2, keep)
        tensors.append(tensor)

        remaining = np.diag(S) @ Vh
        left_bond = keep

    last_tensor = remaining.reshape(left_bond, 2, 1)
    tensors.append(last_tensor)

    return MatrixProductState(tensors)
