import networkx as nx
import torch

from qiskit import QuantumCircuit
from torch_geometric.nn import GCNConv, global_mean_pool


NUM_QUBITS = 5

LOOKAHEAD_GATES = 6

MAX_SWAPS_PER_GATE = 10

GNN_WEIGHT = 0.35
LOOKAHEAD_WEIGHT = 0.65

SAFETY_THRESHOLD = 1.5


class GNNRouter(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = GCNConv(9, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 16)

        self.fc1 = torch.nn.Linear(16, 16)
        self.fc2 = torch.nn.Linear(16, 1)

        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index, batch):

        x = self.conv1(x, edge_index)
        x = self.relu(x)

        x = self.conv2(x, edge_index)
        x = self.relu(x)

        x = self.conv3(x, edge_index)
        x = self.relu(x)

        x = global_mean_pool(x, batch)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)

        return x.squeeze(-1)


model = GNNRouter()

model.load_state_dict(
    torch.load(
        "gnn_routing_model.pt",
        map_location="cpu"
    )
)

model.eval()


def swap_layout(layout, a, b):

    new_layout = layout.copy()

    for virtual_q, physical_q in layout.items():

        if physical_q == a:
            new_layout[virtual_q] = b

        elif physical_q == b:
            new_layout[virtual_q] = a

    return new_layout


def create_edge_index(graph):

    edges = []

    for a, b in graph.edges():

        edges.append([a, b])
        edges.append([b, a])

    return torch.tensor(
        edges,
        dtype=torch.long
    ).t().contiguous()


def create_features(
    graph,
    layout,
    q0,
    q1,
    swap_a,
    swap_b,
    upcoming,
    remaining,
    pressure
):

    p0 = layout[q0]
    p1 = layout[q1]

    features = []

    for node in range(NUM_QUBITS):

        q0_flag = int(node == p0)
        q1_flag = int(node == p1)

        swap_a_flag = int(node == swap_a)
        swap_b_flag = int(node == swap_b)

        d0 = nx.shortest_path_length(
            graph,
            node,
            p0
        )

        d1 = nx.shortest_path_length(
            graph,
            node,
            p1
        )

        features.append([
            q0_flag,
            q1_flag,
            swap_a_flag,
            swap_b_flag,
            d0,
            d1,
            upcoming,
            remaining,
            pressure
        ])

    return torch.tensor(
        features,
        dtype=torch.float
    )


def gnn_predictions(
    graph,
    layout,
    q0,
    q1,
    swap_options,
    upcoming,
    remaining,
    pressure
):

    if not swap_options:
        return []

    edge_index = create_edge_index(graph)

    all_x = []
    all_batch = []
    all_edges = []

    for index, (a, b) in enumerate(
        swap_options
    ):

        x = create_features(
            graph,
            layout,
            q0,
            q1,
            a,
            b,
            upcoming,
            remaining,
            pressure
        )

        all_x.append(x)

        all_batch.append(
            torch.full(
                (NUM_QUBITS,),
                index,
                dtype=torch.long
            )
        )

        all_edges.append(
            edge_index + index * NUM_QUBITS
        )

    x = torch.cat(
        all_x,
        dim=0
    )

    batch = torch.cat(
        all_batch,
        dim=0
    )

    edge_index = torch.cat(
        all_edges,
        dim=1
    )

    with torch.no_grad():

        predictions = model(
            x,
            edge_index,
            batch
        )

    return predictions.tolist()


def future_cost(
    graph,
    layout,
    gates,
    start_index
):

    cost = 0

    end = min(
        len(gates),
        start_index + LOOKAHEAD_GATES
    )

    for i in range(
        start_index,
        end
    ):

        q0, q1 = gates[i]

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


def immediate_cost(
    graph,
    layout,
    q0,
    q1
):

    p0 = layout[q0]
    p1 = layout[q1]

    return nx.shortest_path_length(
        graph,
        p0,
        p1
    ) - 1


def choose_swap(
    graph,
    layout,
    q0,
    q1,
    options,
    gates,
    gate_index,
    upcoming,
    remaining,
    pressure
):

    if not options:
        return None

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

    candidates = []

    for i, swap in enumerate(options):

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

        total_cost = (
            current_cost
            +
            lookahead
        )

        candidates.append({
            "swap": swap,
            "gnn": predictions[i],
            "lookahead": lookahead,
            "total": total_cost
        })

    total_values = [
        x["total"]
        for x in candidates
    ]

    gnn_values = [
        x["gnn"]
        for x in candidates
    ]

    min_total = min(
        total_values
    )

    max_total = max(
        total_values
    )

    min_gnn = min(
        gnn_values
    )

    max_gnn = max(
        gnn_values
    )

    for item in candidates:

        if max_total > min_total:

            routing_score = (
                item["total"] - min_total
            ) / (
                max_total - min_total
            )

        else:

            routing_score = 0

        if max_gnn > min_gnn:

            gnn_score = (
                item["gnn"] - min_gnn
            ) / (
                max_gnn - min_gnn
            )

        else:

            gnn_score = 0

        item["score"] = (
            LOOKAHEAD_WEIGHT * routing_score
            +
            GNN_WEIGHT * gnn_score
        )

    best_combined = min(
        candidates,
        key=lambda x: x["score"]
    )

    safest = min(
        candidates,
        key=lambda x: x["total"]
    )

    if (
        best_combined["total"]
        >
        safest["total"] + SAFETY_THRESHOLD
    ):

        return safest["swap"]

    return best_combined["swap"]


def ml_route_circuit(
    circuit,
    coupling_map
):

    graph = nx.Graph()

    for edge in coupling_map.get_edges():

        graph.add_edge(
            edge[0],
            edge[1]
        )

    layout = {
        i: i
        for i in range(
            circuit.num_qubits
        )
    }

    routed_qc = QuantumCircuit(
        circuit.num_qubits
    )

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

    swap_count = 0

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
            for q in range(
                circuit.num_qubits
            )
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

        swaps_this_gate = 0

        while not graph.has_edge(
            p0,
            p1
        ):

            if swaps_this_gate >= MAX_SWAPS_PER_GATE:

                path = nx.shortest_path(
                    graph,
                    p0,
                    p1
                )

                for i in range(
                    len(path) - 2
                ):

                    a = path[i]
                    b = path[i + 1]

                    routed_qc.swap(
                        a,
                        b
                    )

                    layout = swap_layout(
                        layout,
                        a,
                        b
                    )

                    swap_count += 1
                    swaps_this_gate += 1

                p0 = layout[q0]
                p1 = layout[q1]

                break

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

            options = set()

            for i in range(
                len(path) - 1
            ):

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

                options = improving

            else:

                options = list(options)

            best_swap = choose_swap(
                graph,
                layout,
                q0,
                q1,
                options,
                gates,
                gate_index,
                upcoming,
                remaining,
                pressure
            )

            if best_swap is None:
                break

            a, b = best_swap

            routed_qc.swap(
                a,
                b
            )

            layout = swap_layout(
                layout,
                a,
                b
            )

            swap_count += 1
            swaps_this_gate += 1

            p0 = layout[q0]
            p1 = layout[q1]

        if graph.has_edge(
            p0,
            p1
        ):

            routed_qc.append(
                gate,
                [p0, p1]
            )

        gate_index += 1

    return routed_qc, swap_count


def demonstrate_gnn_prediction(
    circuit,
    coupling_map
):

    graph = nx.Graph()

    for edge in coupling_map.get_edges():

        graph.add_edge(
            edge[0],
            edge[1]
        )

    layout = {
        i: i
        for i in range(
            circuit.num_qubits
        )
    }

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

    print()
    print("========================================")
    print("       GNN PREDICTION DEMONSTRATION")
    print("========================================")

    demonstrated = False

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

        options = set()

        for i in range(
            len(path) - 1
        ):

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

        if not options:
            continue

        upcoming = max(
            0,
            len(gates) - gate_index - 1
        )

        remaining = (
            len(gates) - gate_index
        )

        interaction_count = {
            q: 0
            for q in range(
                circuit.num_qubits
            )
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

        options = list(options)

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

        print()
        print(
            f"Current gate: CX(q{q0}, q{q1})"
        )

        print(
            f"Current mapping: "
            f"q{q0}->p{p0}, "
            f"q{q1}->p{p1}"
        )

        print()
        print(
            "Candidate SWAPs and GNN predictions"
        )

        print(
            "----------------------------------------"
        )

        best_index = min(
            range(
                len(predictions)
            ),
            key=lambda i: predictions[i]
        )

        for i, swap in enumerate(options):

            print(
                f"SWAP {swap} "
                f"-> Predicted future cost: "
                f"{predictions[i]:.2f}"
            )

        best_swap = options[best_index]

        print(
            "----------------------------------------"
        )

        print(
            f"GNN selected SWAP: {best_swap}"
        )

        print(
            f"Predicted future cost: "
            f"{predictions[best_index]:.2f}"
        )

        test_layout = swap_layout(
            layout,
            best_swap[0],
            best_swap[1]
        )

        actual_cost = future_cost(
            graph,
            test_layout,
            gates,
            gate_index + 1
        )

        prediction_error = abs(
            predictions[best_index]
            - actual_cost
        )

        print(
            f"Actual future routing cost: "
            f"{actual_cost}"
        )

        print(
            f"Prediction error: "
            f"{prediction_error:.2f}"
        )

        print()
        print(
            "The GNN predicted the future "
            "routing cost before the SWAP "
            "was applied."
        )

        print(
            "========================================"
        )

        demonstrated = True
        break

    if not demonstrated:

        print()
        print(
            "No non-adjacent gate was found "
            "for prediction demonstration."
        )

    print()