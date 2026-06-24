"""
Bond-3 G-matrix construction.

D = 3 means:
    physical dimension = 2
    virtual bond dimension = 3

So native G has dimension:
    2 * 3 = 6

Qiskit needs power-of-two dimensions, so:
    6x6 native G -> 8x8 three-qubit gate
"""

import numpy as np
import scipy.linalg
import quimb.tensor as qtn

from qiskit.circuit.library import UnitaryGate


def bond3_mps_approximation(psi):
    """
    Create a right-canonical bond-3 MPS approximation.
    """
    psi = np.asarray(psi, dtype=complex)

    norm = np.linalg.norm(psi)

    if not np.isclose(norm, 1.0):
        raise ValueError(f"Statevector must be normalized. Norm = {norm}")

    mps = qtn.MatrixProductState.from_dense(
        psi,
        max_bond=3,
        absorb="left",
    )

    mps.normalize()
    mps.right_canonicalize(inplace=True)
    mps.permute_arrays(shape="lpr")

    return mps


def quimb_bond_dimensions(mps):
    """
    Return bond dimensions of a Quimb MPS.
    """
    bonds = []

    for i in range(mps.num_tensors - 1):
        left_inds = set(mps[i].inds)
        right_inds = set(mps[i + 1].inds)

        shared = list(left_inds.intersection(right_inds))

        if len(shared) != 1:
            raise ValueError(
                f"Could not identify bond between site {i} and site {i + 1}"
            )

        bond_ind = shared[0]
        bonds.append(mps.ind_size(bond_ind))

    return bonds


def is_unitary(U, atol=1e-8):
    """
    Check whether a matrix is unitary.
    """
    U = np.asarray(U, dtype=complex)

    if U.shape[0] != U.shape[1]:
        return False

    eye = np.eye(U.shape[0], dtype=complex)

    return np.allclose(U.conj().T @ U, eye, atol=atol)


def complete_columns_to_unitary(columns, atol=1e-10):
    """
    Complete an isometry into a full unitary.

    Example:
        D = 3 middle tensor gives columns of shape 6x3.
        We complete 6x3 -> 6x6.
    """
    columns = np.asarray(columns, dtype=complex)

    if columns.ndim != 2:
        raise ValueError("columns must be a 2D matrix.")

    native_dim, num_cols = columns.shape

    if num_cols > native_dim:
        raise ValueError(
            f"Cannot complete {native_dim}x{num_cols} matrix to a unitary."
        )

    gram = columns.conj().T @ columns

    if not np.allclose(gram, np.eye(num_cols), atol=atol):
        eigvals, eigvecs = np.linalg.eigh(gram)

        if np.any(eigvals < atol):
            raise ValueError("Columns are linearly dependent.")

        inv_sqrt = (
            eigvecs
            @ np.diag(1.0 / np.sqrt(eigvals))
            @ eigvecs.conj().T
        )

        columns = columns @ inv_sqrt

    null_space = scipy.linalg.null_space(columns.conj().T)

    U = np.column_stack((columns, null_space))

    if U.shape != (native_dim, native_dim):
        raise ValueError("Failed to complete matrix to square unitary.")

    if not is_unitary(U):
        raise ValueError("Completed matrix is not unitary.")

    return U


def tensor_to_lpr_3d(tensor_data, site, n_sites):
    """
    Convert Quimb tensor into shape:

        left_bond, physical, right_bond
    """
    A = np.asarray(tensor_data, dtype=complex)

    if A.ndim == 3:
        return A

    if A.ndim == 2:
        if site == 0:
            # first tensor: physical, right
            if A.shape[0] != 2:
                raise ValueError(f"Unexpected first tensor shape: {A.shape}")

            return A.reshape(1, A.shape[0], A.shape[1])

        if site == n_sites - 1:
            # last tensor: left, physical
            if A.shape[1] != 2:
                raise ValueError(f"Unexpected last tensor shape: {A.shape}")

            return A.reshape(A.shape[0], A.shape[1], 1)

    raise ValueError(f"Unsupported tensor shape at site {site}: {A.shape}")


def swap_physical_and_bond(right_dim):
    """
    Generalized SWAP between physical qubit dimension 2
    and virtual bond dimension right_dim.

    For D = 3, this is a 6x6 qubit-qutrit swap.
    """
    physical_dim = 2
    native_dim = physical_dim * right_dim

    S = np.zeros((native_dim, native_dim), dtype=complex)

    for p in range(physical_dim):
        for b in range(right_dim):
            old_index = p * right_dim + b
            new_index = b * physical_dim + p

            S[new_index, old_index] = 1.0

    return S


def next_power_of_two(dim):
    """
    Smallest power of two >= dim.
    """
    return 1 << int(np.ceil(np.log2(dim)))


def embed_native_unitary_to_qubit_unitary(U_native):
    """
    Embed native unitary into qubit Hilbert space.

    Examples:
        2x2 -> 2x2
        4x4 -> 4x4
        6x6 -> 8x8
    """
    U_native = np.asarray(U_native, dtype=complex)

    if U_native.shape[0] != U_native.shape[1]:
        raise ValueError("Native unitary must be square.")

    if not is_unitary(U_native):
        raise ValueError("Native matrix is not unitary.")

    native_dim = U_native.shape[0]
    qubit_dim = next_power_of_two(native_dim)

    U_qubit = np.eye(qubit_dim, dtype=complex)
    U_qubit[:native_dim, :native_dim] = U_native

    if not is_unitary(U_qubit):
        raise ValueError("Embedded qubit matrix is not unitary.")

    return U_qubit


def bond3_G_matrices(mps):
    """
    Construct D=3 G matrices from a bond-3 MPS.

    Middle tensors give native 6x6 gates.
    These are embedded into 8x8 three-qubit Qiskit gates.
    """
    max_bond = mps.max_bond()

    if max_bond is None:
        raise ValueError("One-site MPS is not supported yet.")

    if max_bond > 3:
        raise ValueError("MPS bond dimension must be <= 3.")

    n_sites = mps.num_tensors

    G_data = []

    for site in range(n_sites):
        A = tensor_to_lpr_3d(
            mps[site].data.copy(),
            site=site,
            n_sites=n_sites,
        )

        left_dim, physical_dim, right_dim = A.shape

        if physical_dim != 2:
            raise ValueError("Only physical qubits are supported.")

        native_dim = physical_dim * right_dim

        columns = []

        for left_index in range(left_dim):
            col = A[left_index, :, :].reshape(native_dim, 1)

            if not np.allclose(col, 0.0):
                columns.append(col)

        if len(columns) == 0:
            raise ValueError(f"No nonzero columns at site {site}.")

        incomplete = np.column_stack(columns)

        U_native = complete_columns_to_unitary(incomplete)

        # BlackWild-style convention:
        # middle tensors need physical-bond swap.
        if site != 0 and right_dim > 1:
            S = swap_physical_and_bond(right_dim)
            U_native = U_native @ S

        if not is_unitary(U_native):
            raise ValueError(f"Native G at site {site} is not unitary.")

        U_qubit = embed_native_unitary_to_qubit_unitary(U_native)

        num_qubits = int(np.log2(U_qubit.shape[0]))

        gate = UnitaryGate(
            U_qubit,
            label=f"D3G{site}",
        )

        G_data.append(
            {
                "site": site,
                "left_dim": left_dim,
                "right_dim": right_dim,
                "native_dim": native_dim,
                "qubit_dim": U_qubit.shape[0],
                "num_qubits": num_qubits,
                "native_unitary": U_native,
                "qubit_unitary": U_qubit,
                "qiskit_gate": gate,
                "embedded": native_dim != U_qubit.shape[0],
            }
        )

    return G_data
