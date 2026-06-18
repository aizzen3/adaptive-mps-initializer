def decompose_circuit(qc, reps=10):
    """
    Decompose high-level circuit blocks into lower-level gates.
    """
    decomposed = qc

    for _ in range(reps):
        decomposed = decomposed.decompose()

    return decomposed


def circuit_gate_report(qc, decompose=True, reps=10):
    """
    Return simple circuit metrics.

    If Qiskit fails while decomposing a generic unitary block, return
    a safe error report instead of crashing the whole benchmark.
    """

    try:
        if decompose:
            work_qc = decompose_circuit(qc, reps=reps)
        else:
            work_qc = qc

        return {
            "num_qubits": work_qc.num_qubits,
            "depth": work_qc.depth(),
            "size": work_qc.size(),
            "gate_counts": dict(work_qc.count_ops()),
            "decomposition_error": None,
        }

    except BaseException as exc:
        return {
            "num_qubits": qc.num_qubits,
            "depth": None,
            "size": None,
            "gate_counts": dict(qc.count_ops()),
            "decomposition_error": repr(exc),
        }
