import random
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from router import route_circuit_baseline
from gnn_router import ml_route_circuit as gnn_route_circuit

# Define 4 distinct, heterogeneous quantum topologies
TOPOLOGIES = {
    "5-Qubit Line (Trained)": {
        "num_qubits": 5,
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
        "desc": "Baseline Linear Path"
    },
    "5-Qubit T-Junction (Zero-Shot)": {
        "num_qubits": 5,
        "edges": [[0, 1], [1, 2], [2, 3], [1, 4]],
        "desc": "Branching Junction Topology"
    },
    "5-Qubit Cyclic Ring (Zero-Shot)": {
        "num_qubits": 5,
        "edges": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]],
        "desc": "Cyclic Boundary Loop"
    },
    "7-Qubit IBM Falcon Subgraph (Zero-Shot)": {
        "num_qubits": 7,
        "edges": [[0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [5, 6]],
        "desc": "Heterogeneous 7-Qubit Tree"
    }
}

CIRCUITS_PER_TOPOLOGY = 50
GATES_RANGE = (30, 40)

results = []

print("=" * 80)
print("ZERO-SHOT CROSS-TOPOLOGY BENCHMARK (30-40 GATE DEEP CIRCUITS)")
print("Evaluating trained GNN without retraining across 4 heterogeneous topologies")
print("=" * 80)

for topo_name, topo_info in TOPOLOGIES.items():
    n_qubits = topo_info["num_qubits"]
    edges = topo_info["edges"]
    coupling_map = CouplingMap(edges)

    print(f"\n[+] Testing {topo_name} ({topo_info['desc']}) over {CIRCUITS_PER_TOPOLOGY} deep circuits...")

    baseline_total_swaps = 0
    gnn_total_swaps = 0

    random.seed(42)

    for c_idx in range(CIRCUITS_PER_TOPOLOGY):
        num_gates = random.randint(*GATES_RANGE)
        qc = QuantumCircuit(n_qubits)

        for _ in range(num_gates):
            q0, q1 = random.sample(range(n_qubits), 2)
            qc.cx(q0, q1)

        _, base_swaps = route_circuit_baseline(qc, coupling_map)
        _, gnn_swaps = gnn_route_circuit(qc, coupling_map)

        baseline_total_swaps += base_swaps
        gnn_total_swaps += gnn_swaps

    base_avg = baseline_total_swaps / CIRCUITS_PER_TOPOLOGY
    gnn_avg = gnn_total_swaps / CIRCUITS_PER_TOPOLOGY
    reduction = ((baseline_total_swaps - gnn_total_swaps) / baseline_total_swaps) * 100

    results.append({
        "Topology": topo_name,
        "Qubits": n_qubits,
        "Coupling Type": topo_info["desc"],
        "Baseline Avg SWAPs": round(base_avg, 2),
        "GNN Avg SWAPs": round(gnn_avg, 2),
        "SWAP Reduction (%)": f"+{reduction:.2f}%"
    })

    print(f"    Baseline Avg: {base_avg:.2f} | GNN Avg: {gnn_avg:.2f} | Net Reduction: +{reduction:.2f}%")

df_results = pd.DataFrame(results)

print("\n" + "=" * 80)
print("ZERO-SHOT TRANSFER LEARNING SUMMARY")
print("=" * 80)
print(df_results.to_string(index=False))
print("=" * 80)

df_results.to_csv("results/multitopology_results.csv", index=False)
print("Saved benchmark results to results/multitopology_results.csv")
