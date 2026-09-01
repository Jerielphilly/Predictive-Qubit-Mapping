import random
import networkx as nx
import pandas as pd


NUM_CIRCUITS = 5000
NUM_QUBITS = 5


graph = nx.path_graph(NUM_QUBITS)


def swap_layout(layout, a, b):

    new_layout = layout.copy()

    for virtual_q, physical_q in layout.items():

        if physical_q == a:
            new_layout[virtual_q] = b

        elif physical_q == b:
            new_layout[virtual_q] = a

    return new_layout


def routing_cost(graph, layout, gates):

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


print("Generating improved dataset...")

rows = []


for circuit_number in range(NUM_CIRCUITS):

    random.seed(circuit_number)

    num_gates = random.randint(5, 15)

    gates = []

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

        gates.append(
            (q0, q1)
        )


    layout = {
        i: i
        for i in range(NUM_QUBITS)
    }


    current_swap_count = 0


    for gate_index, (q0, q1) in enumerate(gates):

        p0 = layout[q0]
        p1 = layout[q1]


        if graph.has_edge(
            p0,
            p1
        ):

            continue


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


        remaining_gates = gates[
            gate_index:
        ]

        future_gates = gates[
            gate_index + 1:
        ]


        for swap_a, swap_b in candidates:

            test_layout = swap_layout(
                layout,
                swap_a,
                swap_b
            )


            new_p0 = test_layout[q0]
            new_p1 = test_layout[q1]


            current_distance = nx.shortest_path_length(
                graph,
                p0,
                p1
            )


            new_distance = nx.shortest_path_length(
                graph,
                new_p0,
                new_p1
            )


            future_cost = routing_cost(
                graph,
                test_layout,
                future_gates
            )


            total_cost = (
                max(0, new_distance - 1)
                + future_cost
            )


            rows.append({

                "distance":
                    current_distance,

                "upcoming_gates":
                    len(future_gates),

                "remaining_gates":
                    len(remaining_gates),

                "physical_q0":
                    p0,

                "physical_q1":
                    p1,

                "current_swap_count":
                    current_swap_count,

                "candidate_swap_a":
                    swap_a,

                "candidate_swap_b":
                    swap_b,

                "future_swap_cost":
                    total_cost
            })


        # Apply the best immediate baseline SWAP
        # only to continue generating realistic states.

        best_swap = None
        best_cost = float("inf")


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


            future_cost = routing_cost(
                graph,
                test_layout,
                future_gates
            )


            total_cost = (
                max(0, new_distance - 1)
                + future_cost
            )


            if total_cost < best_cost:

                best_cost = total_cost

                best_swap = (
                    swap_a,
                    swap_b
                )


        if best_swap is not None:

            swap_a, swap_b = best_swap

            layout = swap_layout(
                layout,
                swap_a,
                swap_b
            )

            current_swap_count += 1


df = pd.DataFrame(rows)


df.to_csv(
    "dataset.csv",
    index=False
)


print()
print("Dataset generation complete!")
print("Circuits generated:", NUM_CIRCUITS)
print("Training samples:", len(df))
print("Saved as: dataset.csv")