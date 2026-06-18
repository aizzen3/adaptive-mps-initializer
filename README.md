# Adaptive MPS Initializer

Adaptive MPS Initializer is a research package for preparing quantum states from arbitrary 1D functions using matrix product states.

The package can:

- sample an arbitrary function
- convert it into a quantum statevector
- decompose the statevector into MPS tensors
- find the smallest bond dimension for a target fidelity
- build a Qiskit circuit from the compressed MPS
- compare MPS circuits with dense Qiskit StatePreparation
- validate MPS tensors with Quimb

---

## Main idea

Instead of preparing a full dense statevector directly, we use the structure of the state.

Workflow:

```text
function f(x)
    -> sampled values
    -> quantum statevector
    -> MPS decomposition
    -> adaptive compression
    -> Qiskit circuit
