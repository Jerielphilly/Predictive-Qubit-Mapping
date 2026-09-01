import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

def route_circuit_baseline(circuit: QuantumCircuit, coupling_map: CouplingMap) -> tuple[QuantumCircuit, int]:
    # 1. Converting Qiskit CouplingMap to a NetworkX Graph
    G = nx.Graph()
    for edge in coupling_map.get_edges():
        G.add_edge(edge[0], edge[1])

    routed_qc = QuantumCircuit(circuit.num_qubits)
    
    # 2. Tracking virtual qubits to physical qubits
    layout = {i: i for i in range(circuit.num_qubits)}
    swap_count = 0

    # 3. Processing every instruction in the original circuit
    for instruction in circuit.data:
        gate = instruction.operation
        qubits = instruction.qubits
        
        # Single-qubit gates pass through normally
        if len(qubits) == 1:
            phys_q = layout[circuit.find_bit(qubits[0]).index]
            routed_qc.append(gate, [phys_q])

        # Two-qubit gates needing routing validation
        elif len(qubits) == 2:
            v_q0 = circuit.find_bit(qubits[0]).index
            v_q1 = circuit.find_bit(qubits[1]).index
            
            p_q0 = layout[v_q0]
            p_q1 = layout[v_q1]

            # If already connected, just execute the gate
            if G.has_edge(p_q0, p_q1):
                routed_qc.append(gate, [p_q0, p_q1])
            else:
                # 4. Routing: Find shortest path using NetworkX
                path = nx.shortest_path(G, source=p_q0, target=p_q1)
                
                # Insert SWAPs to move data
                for i in range(len(path) - 2):
                    node1 = path[i]
                    node2 = path[i + 1]
                    routed_qc.swap(node1, node2)
                    swap_count += 1
                    
                    # Update layout tracking map
                    for v_q, p_q in layout.items():
                        if p_q == node1:
                            layout[v_q] = node2
                        elif p_q == node2:
                            layout[v_q] = node1

                # Execute original gate since adjacent
                final_q0 = layout[v_q0]
                final_q1 = layout[v_q1]
                routed_qc.append(gate, [final_q0, final_q1])

    return routed_qc, swap_count