import random
import torch
import networkx as nx

from torch_geometric.data import Data


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

        if not graph.has_edge(p0, p1):

            distance = nx.shortest_path_length(
                graph,
                p0,
                p1
            )

            cost += distance - 1

    return cost


def create_node_features(
    layout,
    q0,
    q1,
    swap_a,
    swap_b,
    upcoming_gates,
    remaining_gates,
    gate_pressure
):

    p0 = layout[q0]
    p1 = layout[q1]

    features = []

    for node in range(NUM_QUBITS):

        is_q0 = int(node == p0)
        is_q1 = int(node == p1)

        is_swap_a = int(node == swap_a)
        is_swap_b = int(node == swap_b)

        distance_q0 = nx.shortest_path_length(
            graph,
            node,
            p0
        )

        distance_q1 = nx.shortest_path_length(
            graph,
            node,
            p1
        )

        features.append([
            is_q0,
            is_q1,
            is_swap_a,
            is_swap_b,
            distance_q0,
            distance_q1,
            upcoming_gates,
            remaining_gates,
            gate_pressure
        ])

    return features


print("Generating improved GNN dataset...")
print()

rows = []


# A fresh random generator.
# No fixed circuit seed is used.
rng = random.SystemRandom()


for circuit_number in range(NUM_CIRCUITS):

    num_gates = rng.randint(
        8,
        20
    )

    gates = []

    for _ in range(num_gates):

        q0 = rng.randrange(
            NUM_QUBITS
        )

        q1 = rng.randrange(
            NUM_QUBITS
        )

        while q1 == q0:

            q1 = rng.randrange(
                NUM_QUBITS
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


        remaining_gates = gates[
            gate_index:
        ]

        future_gates = gates[
            gate_index + 1:
        ]


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


        # Count how often each physical qubit
        # appears in future interactions.
        interaction_count = {
            q: 0
            for q in range(NUM_QUBITS)
        }


        for future_q0, future_q1 in future_gates:

            interaction_count[
                future_q0
            ] += 1

            interaction_count[
                future_q1
            ] += 1


        for swap_a, swap_b in candidates:

            test_layout = swap_layout(
                layout,
                swap_a,
                swap_b
            )


            new_p0 = test_layout[q0]
            new_p1 = test_layout[q1]


            distance = nx.shortest_path_length(
                graph,
                new_p0,
                new_p1
            )


            future_cost = routing_cost(
                graph,
                test_layout,
                future_gates
            )


            # Extra information describing
            # how important the current qubits are
            # in the remaining circuit.
            gate_pressure = (
                interaction_count[q0]
                + interaction_count[q1]
            )


            # The target represents the estimated
            # routing work after choosing this SWAP.
            target = (
                distance - 1
                + future_cost
            )


            node_features = create_node_features(
                layout,
                q0,
                q1,
                swap_a,
                swap_b,
                len(future_gates),
                len(remaining_gates),
                gate_pressure
            )


            x = torch.tensor(
                node_features,
                dtype=torch.float
            )


            edge_list = []


            for a, b in graph.edges():

                edge_list.append(
                    [a, b]
                )

                edge_list.append(
                    [b, a]
                )


            edge_index = torch.tensor(
                edge_list,
                dtype=torch.long
            ).t().contiguous()


            data = Data(
                x=x,
                edge_index=edge_index,
                y=torch.tensor(
                    [target],
                    dtype=torch.float
                )
            )


            data.swap_a = torch.tensor(
                [swap_a],
                dtype=torch.long
            )

            data.swap_b = torch.tensor(
                [swap_b],
                dtype=torch.long
            )


            dataset_size = len(
                rows
            )


            rows.append(
                data
            )


    if (
        circuit_number + 1
    ) % 500 == 0:

        print(
            "Circuits processed:",
            f"{circuit_number + 1}/{NUM_CIRCUITS}"
        )


torch.save(
    rows,
    "gnn_dataset.pt"
)


print()
print("========================================")
print("       IMPROVED GNN DATASET")
print("========================================")
print(
    "Circuits generated:",
    NUM_CIRCUITS
)
print(
    "Graph samples:",
    len(rows)
)
print(
    "Saved as: gnn_dataset.pt"
)
print()
print("Random generation: ENABLED")
print("Fixed circuit seeds: DISABLED")
print("========================================")