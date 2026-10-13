import random
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from router import route_circuit_baseline
from ml_router import ml_route_circuit as rf_route_circuit
from gnn_router import ml_route_circuit as gnn_route_circuit

topologies = {
    'Linear Path (5Q)': {
        'n': 5,
        'edges': [[0, 1], [1, 2], [2, 3], [3, 4]]
    },
    'T-Junction (5Q)': {
        'n': 5,
        'edges': [[0, 1], [1, 2], [2, 3], [1, 4]]
    },
    'Cyclic Ring (5Q)': {
        'n': 5,
        'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]
    },
    'IBM Falcon (7Q)': {
        'n': 7,
        'edges': [[0, 1], [1, 2], [2, 3], [1, 4], [4, 5], [5, 6]]
    }
}

n_circuits = 50
random.seed(123)

print("=" * 80)
print("DEEP CIRCUIT STRESS TEST (30-40 GATES): GNN vs RANDOM FOREST vs BASELINE")
print("=" * 80)

for name, info in topologies.items():
    n = info['n']
    cmap = CouplingMap(info['edges'])
    b_sw, g_sw, rf_sw = 0, 0, 0
    
    for _ in range(n_circuits):
        num_gates = random.randint(30, 40)
        qc = QuantumCircuit(n)
        for _ in range(num_gates):
            q0, q1 = random.sample(range(n), 2)
            qc.cx(q0, q1)
            
        _, b = route_circuit_baseline(qc, cmap)
        _, g = gnn_route_circuit(qc, cmap)
        _, rf = rf_route_circuit(qc, cmap)
        
        b_sw += b
        g_sw += g
        rf_sw += rf

    base_avg = b_sw / n_circuits
    gnn_avg = g_sw / n_circuits
    rf_avg = rf_sw / n_circuits
    
    winner = "GNN" if gnn_avg < rf_avg else ("RF" if rf_avg < gnn_avg else "Tie")
    print(f"[{name}] Baseline: {base_avg:.2f} | RF: {rf_avg:.2f} | GNN: {gnn_avg:.2f} | Winner: {winner}")
