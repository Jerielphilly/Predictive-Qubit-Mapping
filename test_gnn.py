import random

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from router import route_circuit_baseline
from gnn_router import (
    ml_route_circuit,
    demonstrate_gnn_prediction
)


NUM_CIRCUITS = 20
NUM_QUBITS = 5


coupling_map = CouplingMap([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4]
])


print("========================================")
print("        PHASE 3 GNN TEST")
print("========================================")
print()
print("Testing 20 fresh random circuits...")
print("Fixed seeds: DISABLED")
print()


demo_circuit = QuantumCircuit(
    NUM_QUBITS
)

demo_gates = random.randint(
    5,
    15
)

for _ in range(demo_gates):

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

    demo_circuit.cx(
        q0,
        q1
    )


demonstrate_gnn_prediction(
    demo_circuit,
    coupling_map
)


phase1_total = 0
phase3_total = 0

phase3_better = 0
phase1_better = 0
same_result = 0


for circuit_number in range(
    NUM_CIRCUITS
):

    circuit = QuantumCircuit(
        NUM_QUBITS
    )

    num_gates = random.randint(
        5,
        15
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

        circuit.cx(
            q0,
            q1
        )


    _, phase1_swaps = route_circuit_baseline(
        circuit,
        coupling_map
    )


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


    print(
        f"Circuit {circuit_number + 1:2d}: "
        f"Phase 1 = {phase1_swaps:2d} | "
        f"Phase 3 = {phase3_swaps:2d}"
    )


phase1_average = (
    phase1_total
    /
    NUM_CIRCUITS
)


phase3_average = (
    phase3_total
    /
    NUM_CIRCUITS
)


reduction = (
    (
        phase1_total
        -
        phase3_total
    )
    /
    phase1_total
) * 100 if phase1_total > 0 else 0


print()
print("========================================")
print("              RESULTS")
print("========================================")
print()

print(
    "Circuits tested       :",
    NUM_CIRCUITS
)

print(
    "Phase 1 total SWAPs   :",
    phase1_total
)

print(
    "Phase 3 total SWAPs   :",
    phase3_total
)

print()

print(
    f"Phase 1 average SWAPs : "
    f"{phase1_average:.2f}"
)

print(
    f"Phase 3 average SWAPs : "
    f"{phase3_average:.2f}"
)

print()

print(
    "Phase 3 better        :",
    phase3_better
)

print(
    "Phase 1 better        :",
    phase1_better
)

print(
    "Same result           :",
    same_result
)

print()

if phase3_total < phase1_total:

    print(
        "Phase 3 performed better."
    )

elif phase1_total < phase3_total:

    print(
        "Phase 1 performed better."
    )

else:

    print(
        "Both phases performed the same."
    )


print(
    f"SWAP reduction        : "
    f"{reduction:.2f}%"
)

print()
print("========================================")
print("Fresh random circuits were used.")
print("Fixed seeds: DISABLED")
print("========================================")