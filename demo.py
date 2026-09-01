from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap
from router import route_circuit_baseline

def generate_test_circuit():
    # Build a 5-qubit test circuit 
    qc = QuantumCircuit(5)
    qc.h(0)
    qc.cx(0, 4) 
    qc.cx(1, 3) 
    qc.cx(2, 4) 
    return qc

def main():
    print(" Phase 1: A* Baseline Router Initialization \n")
    
    # 1. Setup Input Circuit
    qc = generate_test_circuit()
    print("1. ORIGINAL CIRCUIT (Software Ideal):")
    print(qc.draw(output='text'))
    print(f"Original Circuit Depth: {qc.depth()}\n")
    
    # 2. Define Hardware (Line Topology: 0-1-2-3-4)
    cmap = CouplingMap([(0,1), (1,2), (2,3), (3,4)])
    print("2. HARDWARE TOPOLOGY:")
    print("Linear Array: [0] <-> [1] <-> [2] <-> [3] <-> [4]\n")
    
    print("3. ROUTING PROCESS")
    routed_qc, total_swaps = route_circuit_baseline(qc, cmap)
    
    print("\n4. FINAL COMPILED CIRCUIT (Hardware Compliant):")
    print(routed_qc.draw(output='text'))
    
    print("\n PERFORMANCE METRICS ")
    print(f"SWAP Gates Injected: {total_swaps}")
    print(f"Final Circuit Depth: {routed_qc.depth()}")

if __name__ == "__main__":
    main()