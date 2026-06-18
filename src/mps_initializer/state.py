import numpy as np


class QuantumState:
    """
    Simple container for a quantum statevector.

    Example:
        phi = [1, 0, 0, 0]
        state = QuantumState(phi)
    """

    def __init__(self, data, normalize=True):
        self.data = np.asarray(data, dtype=complex)

        if self.data.ndim != 1:
            raise ValueError("Quantum state must be a 1D vector.")

        n = self.data.size

        if n == 0:
            raise ValueError("Quantum state cannot be empty.")

        if not self._is_power_of_two(n):
            raise ValueError("Statevector length must be a power of 2.")

        self.n_qubits = int(np.log2(n))

        if normalize:
            self.normalize()

    def normalize(self):
        norm = np.linalg.norm(self.data)

        if norm == 0:
            raise ValueError("Cannot normalize zero vector.")

        self.data = self.data / norm
        return self

    def norm(self):
        return np.linalg.norm(self.data)

    @staticmethod
    def _is_power_of_two(n):
        return n > 0 and (n & (n - 1)) == 0
