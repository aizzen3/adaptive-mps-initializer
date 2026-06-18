import numpy as np

from .state import QuantumState


def qiskit_available():
    """
    Check whether Qiskit is installed.
    """
    try:
        import qiskit
        return True
    except Exception:
        return False


def statevector_to_qiskit_circuit(data, normalize=True, label="StatePrep"):
    """
    Build a Qiskit circuit that prepares the given statevector.

    This uses Qiskit's dense StatePreparation.
    """

    from qiskit import QuantumCircuit
    from qiskit.circuit.library import StatePreparation

    state = QuantumState(data, normalize=normalize)
    psi = state.data
    n_qubits = state.n_qubits

    qc = QuantumCircuit(n_qubits)

    prep = StatePreparation(psi, label=label)
    qc.append(prep, qc.qubits)

    return qc


def circuit_statevector(qc):
    """
    Simulate a Qiskit circuit and return its final statevector.
    """

    from qiskit.quantum_info import Statevector

    return np.asarray(Statevector.from_instruction(qc).data, dtype=complex)


def _complete_columns_to_unitary(columns, column_indices, dim, tol=1e-10):
    """
    Complete selected columns into a full unitary matrix.

    This version:
    1. Preserves already-orthonormal MPS columns.
    2. Repairs non-orthonormal columns only if needed.
    3. Uses an orthogonal projector to find the remaining columns.
    """

    columns = np.asarray(columns, dtype=complex)
    column_indices = [int(i) for i in column_indices]

    if columns.ndim != 2:
        raise ValueError("columns must be a 2D array.")

    if columns.shape[0] != dim:
        raise ValueError("columns must have shape (dim, k).")

    k = columns.shape[1]

    if k != len(column_indices):
        raise ValueError("Number of columns and column_indices must match.")

    if len(set(column_indices)) != len(column_indices):
        raise ValueError("column_indices must be unique.")

    # Normalize requested columns.
    desired = columns.copy()

    for j in range(k):
        norm = np.linalg.norm(desired[:, j])

        if norm < tol:
            raise ValueError("Cannot complete unitary from zero column.")

        desired[:, j] = desired[:, j] / norm

    # Check if requested columns are already orthonormal.
    gram = desired.conj().T @ desired

    if not np.allclose(gram, np.eye(k), atol=1e-8):
        # Repair only when needed.
        desired, _ = np.linalg.qr(desired, mode="reduced")

    # Projector onto the orthogonal complement of desired columns.
    projector = np.eye(dim, dtype=complex) - desired @ desired.conj().T

    # Numerical symmetrization.
    projector = 0.5 * (projector + projector.conj().T)

    eigvals, eigvecs = np.linalg.eigh(projector)

    # Complement eigenvectors have eigenvalue close to 1.
    complement = eigvecs[:, eigvals > 0.5]

    needed = dim - k

    if complement.shape[1] < needed:
        raise RuntimeError(
            f"Could not find enough complement vectors. "
            f"Needed {needed}, got {complement.shape[1]}."
        )

    complement = complement[:, :needed]

    # Assemble U.
    U = np.zeros((dim, dim), dtype=complex)

    for j, col_idx in enumerate(column_indices):
        U[:, col_idx] = desired[:, j]

    extra_j = 0

    for col_idx in range(dim):
        if col_idx in column_indices:
            continue

        U[:, col_idx] = complement[:, extra_j]
        extra_j += 1

    err = np.linalg.norm(U.conj().T @ U - np.eye(dim))

    if err > 1e-8:
        raise ValueError(f"Input matrix is not unitary enough. Error = {err}")

    return U


def _mps_tensor_to_local_unitary(tensor, bond_dim_power):
    """
    Convert one MPS tensor into a local unitary acting on:

        one physical qubit + bond qubits

    Tensor shape:

        (left_bond, 2, right_bond)
    """

    A = np.asarray(tensor, dtype=complex)

    left_dim, physical_dim, right_dim = A.shape

    if physical_dim != 2:
        raise ValueError("Only qubit MPS tensors with physical dimension 2 are supported.")

    B = int(bond_dim_power)
    dim = 2 * B

    if left_dim > B or right_dim > B:
        raise ValueError("bond_dim_power is too small for this tensor.")

    columns = []
    column_indices = []

    for r in range(right_dim):
        vec = np.zeros(dim, dtype=complex)

        for l in range(left_dim):
            for s in range(2):
                row = 2 * l + s
                vec[row] = A[l, s, r]

        # input column represents |right_bond=r>|physical=0>
        col_idx = 2 * r

        columns.append(vec)
        column_indices.append(col_idx)

    columns = np.stack(columns, axis=1)

    U = _complete_columns_to_unitary(
        columns=columns,
        column_indices=column_indices,
        dim=dim,
    )

    return U


def mps_to_sequential_qiskit_circuit(mps, label_prefix="A"):
    """
    Build a sequential MPS preparation circuit using extra bond qubits.

    Qubit layout:
        physical qubits: 0 ... n-1
        bond qubits:     n ... n+m-1

    The MPS tensors are applied from right to left.
    """

    from qiskit import QuantumCircuit

    n_physical = mps.n_qubits
    max_D = mps.max_bond_dimension()

    if max_D <= 1:
        n_bond = 0
    else:
        n_bond = int(np.ceil(np.log2(max_D)))

    bond_dim_power = 2 ** n_bond

    qc = QuantumCircuit(n_physical + n_bond)

    bond_qubits = list(range(n_physical, n_physical + n_bond))

    for site in reversed(range(n_physical)):
        tensor = mps.tensors[site]

        U = _mps_tensor_to_local_unitary(
            tensor,
            bond_dim_power=bond_dim_power,
        )

        # MPS site 0 is most significant bit.
        # Qiskit qubit 0 is least significant bit.
        physical_qubit = n_physical - 1 - site

        qubits = [physical_qubit] + bond_qubits

        qc.unitary(U, qubits, label=f"{label_prefix}{site}")

    return qc


def physical_statevector_from_mps_circuit(qc, n_physical, normalize=True):
    """
    Extract physical statevector from a circuit with bond ancillas.

    Assumes:
        physical qubits are 0 ... n_physical-1
        bond qubits are higher-index qubits
    """

    full = circuit_statevector(qc)

    physical_dim = 2 ** n_physical

    physical = full[:physical_dim]
    leakage = full[physical_dim:]

    leakage_norm = float(np.linalg.norm(leakage))

    if normalize:
        norm = np.linalg.norm(physical)
        if norm > 0:
            physical = physical / norm

    return physical, leakage_norm
