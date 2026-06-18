import csv
import matplotlib.pyplot as plt


def read_results(filename):
    rows = []

    with open(filename, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(
                {
                    "name": row["name"],
                    "chosen_D": int(row["chosen_D"]),
                    "bond_qubits": int(row["bond_qubits"]),
                    "fidelity": float(row["fidelity"]),
                    "cx_gates": int(row["cx_gates"]),
                    "u_gates": int(row["u_gates"]),
                    "depth": int(row["depth"]),
                    "size": int(row["size"]),
                }
            )

    return rows


def main():
    filename = "api_multiple_functions_results.csv"
    rows = read_results(filename)

    names = [r["name"] for r in rows]
    chosen_D = [r["chosen_D"] for r in rows]
    cx_gates = [r["cx_gates"] for r in rows]
    depth = [r["depth"] for r in rows]
    fidelity = [r["fidelity"] for r in rows]

    # Plot 1: chosen D
    plt.figure()
    plt.bar(names, chosen_D)
    plt.xlabel("Function type")
    plt.ylabel("Chosen bond dimension D")
    plt.title("Adaptive MPS: Chosen Bond Dimension")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("plot_functions_chosen_D.png", dpi=300)

    # Plot 2: CX gates
    plt.figure()
    plt.bar(names, cx_gates)
    plt.xlabel("Function type")
    plt.ylabel("CX gate count")
    plt.title("Adaptive MPS: CX Gates by Function Type")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("plot_functions_cx_gates.png", dpi=300)

    # Plot 3: depth
    plt.figure()
    plt.bar(names, depth)
    plt.xlabel("Function type")
    plt.ylabel("Circuit depth")
    plt.title("Adaptive MPS: Circuit Depth by Function Type")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("plot_functions_depth.png", dpi=300)

    # Plot 4: fidelity
    plt.figure()
    plt.bar(names, fidelity)
    plt.axhline(0.999, linestyle="--", label="Target fidelity = 0.999")
    plt.xlabel("Function type")
    plt.ylabel("Preparation fidelity")
    plt.title("Adaptive MPS: Fidelity by Function Type")
    plt.xticks(rotation=30)
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_functions_fidelity.png", dpi=300)

    print("Saved plots:")
    print("plot_functions_chosen_D.png")
    print("plot_functions_cx_gates.png")
    print("plot_functions_depth.png")
    print("plot_functions_fidelity.png")


if __name__ == "__main__":
    main()
