import random
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from router import route_circuit_baseline
from gnn_router import ml_route_circuit as gnn_route_circuit

NUM_TESTS = 10
CIRCUITS_PER_TEST = 100

coupling_map = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4]])

test_results = []
all_results = []

print("========================================")
print("  BASELINE vs GRAPH NEURAL NETWORK")
print("========================================")

overall_base = 0
overall_gnn = 0

for test_number in range(NUM_TESTS):
    base_total = 0
    gnn_total = 0

    print(f"--- TEST {test_number + 1} ---")

    for circuit_number in range(CIRCUITS_PER_TEST):
        circuit = QuantumCircuit(5)
        num_gates = random.randint(20, 40)

        for _ in range(num_gates):
            q0, q1 = random.sample(range(5), 2)
            circuit.cx(q0, q1)

        _, base_swaps = route_circuit_baseline(circuit, coupling_map)
        _, gnn_swaps = gnn_route_circuit(circuit, coupling_map)

        base_total += base_swaps
        gnn_total += gnn_swaps

        all_results.append({
            "Test": test_number + 1,
            "Circuit": circuit_number + 1,
            "Baseline": base_swaps,
            "GNN": gnn_swaps
        })

    print(f"Baseline: {base_total} | GNN: {gnn_total}")
    overall_base += base_total
    overall_gnn += gnn_total

df_tests = pd.DataFrame([{"Baseline Total": overall_base, "GNN Total": overall_gnn}])
df_all = pd.DataFrame(all_results)
df_tests.to_csv("results/benchmark_test_results.csv", index=False)
df_all.to_csv("results/benchmark_all_results.csv", index=False)

print("\n========================================")
print("         OVERALL PERFORMANCE")
print("========================================")
print(f"Total circuits tested : {NUM_TESTS * CIRCUITS_PER_TEST}")
print(f"Baseline total SWAPs  : {overall_base}")
print(f"GNN SWAPs             : {overall_gnn}")
print("========================================")
print("Files saved: benchmark_test_results.csv, benchmark_all_results.csv")