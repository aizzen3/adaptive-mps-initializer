import csv
import matplotlib.pyplot as plt


def read_csv(filename):
    rows = []

    with open(filename, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(
                {
                    "n_qubits": int(row["n_qubits"]),
                    "chosen_D": int(row["chosen_D"]),
                    "bond_qubits": int(row["bond_qubits"]),
                    "mps_fidelity": float(row["mps_fidelity"]),
                    "mps_cx": int(row["mps_cx"]),
                    "mps_depth": int(row["mps_depth"]),
                    "dense_cx": int(row["dense_cx"]),
                    "dense_depth": int(row["dense_depth"]),
                }
            )

    return rows


def main():
    rows = read_csv("adaptive_benchmark.csv")

    n = [r["n_qubits"] for r in rows]

    mps_cx = [r["mps_cx"] for r in rows]
    dense_cx = [r["dense_cx"] for r in rows]

    mps_depth = [r["mps_depth"] for r in rows]
    dense_depth = [r["dense_depth"] for r in rows]

    chosen_D = [r["chosen_D"] for r in rows]
    fidelity = [r["mps_fidelity"] for r in rows]

    # ------------------------------------------------------------
    # Plot 1: CX gates
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(n, mps_cx, marker="o", label="Adaptive MPS")
    plt.plot(n, dense_cx, marker="s", label="Dense StatePreparation")
    plt.xlabel("Number of qubits")
    plt.ylabel("CX gate count")
    plt.title("CX Gate Count: Adaptive MPS vs Dense StatePreparation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_cx_gates.png", dpi=300)

    # ------------------------------------------------------------
    # Plot 2: Circuit depth
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(n, mps_depth, marker="o", label="Adaptive MPS")
    plt.plot(n, dense_depth, marker="s", label="Dense StatePreparation")
    plt.xlabel("Number of qubits")
    plt.ylabel("Circuit depth")
    plt.title("Circuit Depth: Adaptive MPS vs Dense StatePreparation")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_depth.png", dpi=300)

    # ------------------------------------------------------------
    # Plot 3: Chosen bond dimension
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(n, chosen_D, marker="o")
    plt.xlabel("Number of qubits")
    plt.ylabel("Chosen max bond dimension D")
    plt.title("Adaptive Chosen Bond Dimension")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_chosen_D.png", dpi=300)

    # ------------------------------------------------------------
    # Plot 4: Fidelity
    # ------------------------------------------------------------
    plt.figure()
    plt.plot(n, fidelity, marker="o")
    plt.axhline(0.999, linestyle="--", label="Target fidelity = 0.999")
    plt.xlabel("Number of qubits")
    plt.ylabel("MPS preparation fidelity")
    plt.title("Adaptive MPS Preparation Fidelity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plot_fidelity.png", dpi=300)

    print("Saved plots:")
    print("plot_cx_gates.png")
    print("plot_depth.png")
    print("plot_chosen_D.png")
    print("plot_fidelity.png")


if __name__ == "__main__":
    main()
