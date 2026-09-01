import random
import math
import csv

import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from router import route_circuit_baseline
from gnn_router import (
    gnn_predictions,
    swap_layout,
    immediate_cost,
    future_cost
)


NUM_CIRCUITS = 500
NUM_QUBITS = 5
MIN_GATES = 5
MAX_GATES = 15

LOOKAHEAD_GATES = 6


coupling_map = CouplingMap([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4]
])


graph = nx.Graph()

for edge in coupling_map.get_edges():
    graph.add_edge(edge[0], edge[1])


def generate_random_circuit():

    circuit = QuantumCircuit(NUM_QUBITS)

    num_gates = random.randint(
        MIN_GATES,
        MAX_GATES
    )

    for _ in range(num_gates):

        q0 = random.randint(
            0,
            NUM_QUBITS - 1
        )

        q1 = random.randint(
            0,
            NUM_QUBITS - 1
        )

        while q1 == q0:

            q1 = random.randint(
                0,
                NUM_QUBITS - 1
            )

        circuit.cx(q0, q1)

    return circuit


def get_gates(circuit):

    gates = []

    for instruction in circuit.data:

        if len(instruction.qubits) == 2:

            q0 = circuit.find_bit(
                instruction.qubits[0]
            ).index

            q1 = circuit.find_bit(
                instruction.qubits[1]
            ).index

            gates.append(
                (q0, q1)
            )

    return gates


def get_swap_options(
    graph,
    layout,
    q0,
    q1
):

    p0 = layout[q0]
    p1 = layout[q1]

    path = nx.shortest_path(
        graph,
        p0,
        p1
    )

    options = set()

    for i in range(len(path) - 1):

        options.add(
            tuple(
                sorted(
                    (
                        path[i],
                        path[i + 1]
                    )
                )
            )
        )

    current_distance = nx.shortest_path_length(
        graph,
        p0,
        p1
    )

    improving = []

    for a, b in options:

        test_layout = swap_layout(
            layout,
            a,
            b
        )

        new_p0 = test_layout[q0]
        new_p1 = test_layout[q1]

        new_distance = nx.shortest_path_length(
            graph,
            new_p0,
            new_p1
        )

        if new_distance <= current_distance:
            improving.append(
                (a, b)
            )

    if improving:
        return improving

    return list(options)


def evaluate_predictions():

    prediction_errors = []
    squared_errors = []

    correct_rankings = 0
    total_prediction_decisions = 0

    phase1_total = 0
    phase3_total = 0

    phase3_better = 0
    phase1_better = 0
    same_result = 0

    prediction_rows = []

    for circuit_number in range(
        NUM_CIRCUITS
    ):

        circuit = generate_random_circuit()

        gates = get_gates(circuit)

        if not gates:
            continue

        _, phase1_swaps = route_circuit_baseline(
            circuit,
            coupling_map
        )

        from gnn_router import ml_route_circuit

        _, phase3_swaps = ml_route_circuit(
            circuit,
            coupling_map
        )

        phase1_total += phase1_swaps
        phase3_total += phase3_swaps

        if phase3_swaps < phase1_swaps:

            phase3_better += 1

        elif phase1_swaps < phase3_swaps:

            phase1_better += 1

        else:

            same_result += 1


        layout = {
            i: i
            for i in range(NUM_QUBITS)
        }


        gate_index = 0


        for q0, q1 in gates:

            p0 = layout[q0]
            p1 = layout[q1]

            if graph.has_edge(
                p0,
                p1
            ):

                gate_index += 1
                continue


            options = get_swap_options(
                graph,
                layout,
                q0,
                q1
            )

            if not options:
                gate_index += 1
                continue


            remaining = (
                len(gates)
                - gate_index
            )

            upcoming = max(
                0,
                remaining - 1
            )


            interaction_count = {
                q: 0
                for q in range(NUM_QUBITS)
            }


            for fq0, fq1 in gates[
                gate_index + 1:
            ]:

                interaction_count[fq0] += 1
                interaction_count[fq1] += 1


            pressure = (
                interaction_count[q0]
                +
                interaction_count[q1]
            )


            predictions = gnn_predictions(
                graph,
                layout,
                q0,
                q1,
                options,
                upcoming,
                remaining,
                pressure
            )


            actual_costs = []


            for swap in options:

                a, b = swap

                test_layout = swap_layout(
                    layout,
                    a,
                    b
                )

                current_cost = immediate_cost(
                    graph,
                    test_layout,
                    q0,
                    q1
                )

                lookahead = future_cost(
                    graph,
                    test_layout,
                    gates,
                    gate_index + 1
                )

                actual_costs.append(
                    current_cost + lookahead
                )


            if not predictions:
                gate_index += 1
                continue


            predicted_best = min(
                range(len(predictions)),
                key=lambda i: predictions[i]
            )

            actual_best = min(
                range(len(actual_costs)),
                key=lambda i: actual_costs[i]
            )


            if predicted_best == actual_best:

                correct_rankings += 1


            total_prediction_decisions += 1


            for i in range(
                len(predictions)
            ):

                error = abs(
                    predictions[i]
                    -
                    actual_costs[i]
                )

                prediction_errors.append(
                    error
                )

                squared_errors.append(
                    error * error
                )

                prediction_rows.append({
                    "circuit": circuit_number + 1,
                    "gate": gate_index + 1,
                    "swap": str(options[i]),
                    "predicted_cost": predictions[i],
                    "actual_cost": actual_costs[i],
                    "absolute_error": error
                })


            gate_index += 1


        if (
            (circuit_number + 1) % 50 == 0
        ):

            print(
                f"Circuits evaluated: "
                f"{circuit_number + 1}/{NUM_CIRCUITS}"
            )


    mae = (
        sum(prediction_errors)
        /
        len(prediction_errors)
        if prediction_errors
        else 0
    )


    rmse = math.sqrt(
        sum(squared_errors)
        /
        len(squared_errors)
    ) if squared_errors else 0


    ranking_accuracy = (
        correct_rankings
        /
        total_prediction_decisions
        *
        100
        if total_prediction_decisions
        else 0
    )


    reduction = (
        (phase1_total - phase3_total)
        /
        phase1_total
        *
        100
        if phase1_total > 0
        else 0
    )


    return (
        phase1_total,
        phase3_total,
        phase1_better,
        phase3_better,
        same_result,
        reduction,
        mae,
        rmse,
        ranking_accuracy,
        total_prediction_decisions,
        prediction_rows
    )


print("========================================")
print("       PHASE 3 GNN EVALUATION")
print("========================================")
print()
print(f"Testing {NUM_CIRCUITS} fresh random circuits...")
print("Fixed seeds: DISABLED")
print()
print("Evaluating GNN prediction accuracy...")
print()


results = evaluate_predictions()


(
    phase1_total,
    phase3_total,
    phase1_better,
    phase3_better,
    same_result,
    reduction,
    mae,
    rmse,
    ranking_accuracy,
    total_prediction_decisions,
    prediction_rows
) = results


with open(
    "gnn_prediction_results.csv",
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "circuit",
            "gate",
            "swap",
            "predicted_cost",
            "actual_cost",
            "absolute_error"
        ]
    )

    writer.writeheader()
    writer.writerows(
        prediction_rows
    )


print()
print("========================================")
print("          FINAL GNN EVALUATION")
print("========================================")
print()

print(
    "Circuits tested             :",
    NUM_CIRCUITS
)

print(
    "Prediction decisions        :",
    total_prediction_decisions
)

print()

print(
    f"Prediction MAE              : "
    f"{mae:.4f}"
)

print(
    f"Prediction RMSE             : "
    f"{rmse:.4f}"
)

print(
    f"Best-SWAP prediction accuracy: "
    f"{ranking_accuracy:.2f}%"
)

print()

print(
    "Phase 1 total SWAPs         :",
    phase1_total
)

print(
    "Phase 3 total SWAPs         :",
    phase3_total
)

print()

print(
    f"Phase 1 average SWAPs       : "
    f"{phase1_total / NUM_CIRCUITS:.2f}"
)

print(
    f"Phase 3 average SWAPs       : "
    f"{phase3_total / NUM_CIRCUITS:.2f}"
)

print()

print(
    "Phase 3 better circuits     :",
    phase3_better
)

print(
    "Phase 1 better circuits     :",
    phase1_better
)

print(
    "Same result circuits        :",
    same_result
)

print()

print(
    f"Overall SWAP reduction      : "
    f"{reduction:.2f}%"
)

print()

print("========================================")
print("Prediction results saved as:")
print("gnn_prediction_results.csv")
print()
print("Fresh random circuits were used.")
print("Fixed seeds: DISABLED")
print("========================================")