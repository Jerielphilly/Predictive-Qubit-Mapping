import networkx as nx
import joblib

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap


model = joblib.load("routing_model.pkl")


FEATURES = [
    "distance",
    "upcoming_gates",
    "remaining_gates",
    "physical_q0",
    "physical_q1",
    "current_swap_count",
    "candidate_swap_a",
    "candidate_swap_b"
]


def swap_layout(layout, a, b):

    new_layout = layout.copy()

    for virtual_q, physical_q in layout.items():

        if physical_q == a:
            new_layout[virtual_q] = b

        elif physical_q == b:
            new_layout[virtual_q] = a

    return new_layout


def gate_distance(graph, layout, gate):

    q0, q1 = gate

    p0 = layout[q0]
    p1 = layout[q1]

    return nx.shortest_path_length(
        graph,
        p0,
        p1
    )


def future_cost(graph, layout, gates):

    cost = 0

    for q0, q1 in gates:

        p0 = layout[q0]
        p1 = layout[q1]

        distance = nx.shortest_path_length(
            graph,
            p0,
            p1
        )

        if distance > 1:
            cost += distance - 1

    return cost


def ml_route_circuit(circuit, coupling_map):

    graph = nx.Graph()

    for edge in coupling_map.get_edges():
        graph.add_edge(edge[0], edge[1])


    layout = {
        i: i
        for i in range(circuit.num_qubits)
    }


    routed_qc = QuantumCircuit(
        circuit.num_qubits
    )


    swap_count = 0


    two_qubit_gates = []

    for instruction in circuit.data:

        if len(instruction.qubits) == 2:

            q0 = circuit.find_bit(
                instruction.qubits[0]
            ).index

            q1 = circuit.find_bit(
                instruction.qubits[1]
            ).index

            two_qubit_gates.append(
                (q0, q1)
            )


    gate_index = 0


    for instruction in circuit.data:

        gate = instruction.operation
        qubits = instruction.qubits


        if len(qubits) == 1:

            q = circuit.find_bit(
                qubits[0]
            ).index

            routed_qc.append(
                gate,
                [layout[q]]
            )

            continue


        q0 = circuit.find_bit(
            qubits[0]
        ).index

        q1 = circuit.find_bit(
            qubits[1]
        ).index


        p0 = layout[q0]
        p1 = layout[q1]


        remaining = two_qubit_gates[
            gate_index:
        ]


        upcoming = two_qubit_gates[
            gate_index + 1:
        ]


        swaps_for_gate = 0

        max_swaps = 4


        while not graph.has_edge(p0, p1):

            current_distance = nx.shortest_path_length(
                graph,
                p0,
                p1
            )


            path = nx.shortest_path(
                graph,
                p0,
                p1
            )


            candidates = set()


            for i in range(
                len(path) - 1
            ):

                candidates.add(
                    tuple(
                        sorted(
                            (
                                path[i],
                                path[i + 1]
                            )
                        )
                    )
                )


            candidate_data = []


            for swap_a, swap_b in candidates:

                test_layout = swap_layout(
                    layout,
                    swap_a,
                    swap_b
                )


                new_p0 = test_layout[q0]
                new_p1 = test_layout[q1]


                new_distance = nx.shortest_path_length(
                    graph,
                    new_p0,
                    new_p1
                )


                immediate_cost = (
                    new_distance - 1
                )


                future_distance_cost = 0


                lookahead = upcoming[:4]


                for future_gate in lookahead:

                    fq0, fq1 = future_gate

                    fp0 = test_layout[fq0]
                    fp1 = test_layout[fq1]


                    distance = nx.shortest_path_length(
                        graph,
                        fp0,
                        fp1
                    )


                    if distance > 1:

                        future_distance_cost += (
                            distance - 1
                        )


                candidate_data.append({

                    "swap_a": swap_a,

                    "swap_b": swap_b,

                    "layout": test_layout,

                    "new_p0": new_p0,

                    "new_p1": new_p1,

                    "distance": new_distance,

                    "immediate_cost": immediate_cost,

                    "future_cost":
                        future_distance_cost
                })


            if not candidate_data:

                break


            import pandas as pd


            rows = []


            for candidate in candidate_data:

                rows.append({

                    "distance":
                        candidate["distance"],

                    "upcoming_gates":
                        len(upcoming),

                    "remaining_gates":
                        len(remaining),

                    "physical_q0":
                        candidate["new_p0"],

                    "physical_q1":
                        candidate["new_p1"],

                    "current_swap_count":
                        swap_count,

                    "candidate_swap_a":
                        candidate["swap_a"],

                    "candidate_swap_b":
                        candidate["swap_b"]
                })


            data = pd.DataFrame(
                rows,
                columns=FEATURES
            )


            predictions = model.predict(
                data
            )


            min_prediction = min(
                predictions
            )

            max_prediction = max(
                predictions
            )


            if max_prediction == min_prediction:

                ml_scores = [
                    0
                    for _ in predictions
                ]

            else:

                ml_scores = [

                    (
                        prediction
                        - min_prediction
                    )
                    /
                    (
                        max_prediction
                        - min_prediction
                    )

                    for prediction in predictions
                ]


            final_scores = []


            for i, candidate in enumerate(
                candidate_data
            ):

                ml_score = ml_scores[i]


                immediate_score = (
                    candidate["immediate_cost"]
                )


                future_score = (
                    candidate["future_cost"]
                )


                score = (

                    0.50 * ml_score

                    +

                    0.30 * immediate_score

                    +

                    0.20 * future_score

                )


                final_scores.append(
                    score
                )


            best_index = min(
                range(len(final_scores)),
                key=lambda i:
                    final_scores[i]
            )


            best = candidate_data[
                best_index
            ]


            routed_qc.swap(
                best["swap_a"],
                best["swap_b"]
            )


            layout = best["layout"]


            swap_count += 1
            swaps_for_gate += 1


            p0 = layout[q0]
            p1 = layout[q1]


            if swaps_for_gate >= max_swaps:

                while not graph.has_edge(
                    p0,
                    p1
                ):

                    path = nx.shortest_path(
                        graph,
                        p0,
                        p1
                    )


                    swap_a = path[0]
                    swap_b = path[1]


                    routed_qc.swap(
                        swap_a,
                        swap_b
                    )


                    layout = swap_layout(
                        layout,
                        swap_a,
                        swap_b
                    )


                    swap_count += 1


                    p0 = layout[q0]
                    p1 = layout[q1]


                break


        if graph.has_edge(p0, p1):

            routed_qc.append(
                gate,
                [p0, p1]
            )


        gate_index += 1


    return routed_qc, swap_count