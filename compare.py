import random
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from router import route_circuit_baseline
from ml_router import ml_route_circuit


NUM_TESTS = 10
CIRCUITS_PER_TEST = 100


coupling_map = CouplingMap([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4]
])


test_results = []
all_results = []


print("========================================")
print("      PHASE 1 vs PHASE 2 EVALUATION")
print("========================================")
print()
print("Running 10 tests of 100 random circuits")
print("Total circuits: 1000")
print()


overall_phase1 = 0
overall_phase2 = 0

phase2_wins = 0
phase1_wins = 0
same_tests = 0


for test_number in range(NUM_TESTS):

    phase1_total = 0
    phase2_total = 0

    phase2_better = 0
    phase1_better = 0
    same_result = 0


    print("----------------------------------------")
    print(f"TEST {test_number + 1}")
    print("Running 100 new random circuits...")
    print("----------------------------------------")


    for circuit_number in range(CIRCUITS_PER_TEST):

        circuit = QuantumCircuit(5)

        num_gates = random.randint(5, 15)


        for _ in range(num_gates):

            q0 = random.randint(0, 4)
            q1 = random.randint(0, 4)

            while q1 == q0:

                q1 = random.randint(0, 4)

            circuit.cx(q0, q1)


        _, phase1_swaps = route_circuit_baseline(
            circuit,
            coupling_map
        )


        _, phase2_swaps = ml_route_circuit(
            circuit,
            coupling_map
        )


        phase1_total += phase1_swaps
        phase2_total += phase2_swaps


        if phase2_swaps < phase1_swaps:

            phase2_better += 1

        elif phase1_swaps < phase2_swaps:

            phase1_better += 1

        else:

            same_result += 1


        all_results.append({

            "Test": test_number + 1,

            "Circuit": circuit_number + 1,

            "Phase 1": phase1_swaps,

            "Phase 2": phase2_swaps

        })


    phase1_average = (
        phase1_total / CIRCUITS_PER_TEST
    )


    phase2_average = (
        phase2_total / CIRCUITS_PER_TEST
    )


    if phase1_total > 0:

        reduction = (
            (phase1_total - phase2_total)
            / phase1_total
        ) * 100

    else:

        reduction = 0


    if phase2_total < phase1_total:

        winner = "Phase 2"
        phase2_wins += 1

    elif phase1_total < phase2_total:

        winner = "Phase 1"
        phase1_wins += 1

    else:

        winner = "Same"
        same_tests += 1


    test_results.append({

        "Test": test_number + 1,

        "Circuits": CIRCUITS_PER_TEST,

        "Phase 1 Total SWAPs":
            phase1_total,

        "Phase 2 Total SWAPs":
            phase2_total,

        "Phase 1 Average":
            round(phase1_average, 2),

        "Phase 2 Average":
            round(phase2_average, 2),

        "Reduction (%)":
            round(reduction, 2),

        "Phase 2 Better":
            phase2_better,

        "Phase 1 Better":
            phase1_better,

        "Same":
            same_result,

        "Winner":
            winner

    })


    overall_phase1 += phase1_total
    overall_phase2 += phase2_total


    print()
    print(f"Phase 1 total : {phase1_total}")
    print(f"Phase 2 total : {phase2_total}")
    print(f"Reduction     : {reduction:.2f}%")
    print(f"Phase 2 better: {phase2_better}")
    print(f"Phase 1 better: {phase1_better}")
    print(f"Same          : {same_result}")
    print(f"Winner        : {winner}")
    print()


total_circuits = (
    NUM_TESTS * CIRCUITS_PER_TEST
)


overall_phase1_average = (
    overall_phase1 / total_circuits
)


overall_phase2_average = (
    overall_phase2 / total_circuits
)


if overall_phase1 > 0:

    overall_reduction = (
        (overall_phase1 - overall_phase2)
        / overall_phase1
    ) * 100

else:

    overall_reduction = 0


df_tests = pd.DataFrame(
    test_results
)


df_all = pd.DataFrame(
    all_results
)


df_tests.to_csv(
    "ten_test_results.csv",
    index=False
)


df_all.to_csv(
    "all_circuit_results.csv",
    index=False
)


print()
print("========================================")
print("           FINAL RESULTS")
print("========================================")


print()
print(df_tests.to_string(index=False))


print()
print("========================================")
print("         OVERALL PERFORMANCE")
print("========================================")


print()
print(
    "Total circuits tested :",
    total_circuits
)

print(
    "Number of tests       :",
    NUM_TESTS
)


print()
print(
    f"Phase 1 total SWAPs   : {overall_phase1}"
)

print(
    f"Phase 2 total SWAPs   : {overall_phase2}"
)


print()
print(
    f"Phase 1 average SWAPs : "
    f"{overall_phase1_average:.2f}"
)

print(
    f"Phase 2 average SWAPs : "
    f"{overall_phase2_average:.2f}"
)


print()
print(
    f"Phase 2 won tests     : "
    f"{phase2_wins}"
)

print(
    f"Phase 1 won tests     : "
    f"{phase1_wins}"
)

print(
    f"Same tests            : "
    f"{same_tests}"
)


print()
print(
    f"Overall SWAP reduction: "
    f"{overall_reduction:.2f}%"
)


print()
print("========================================")
print("Files saved:")
print("ten_test_results.csv")
print("all_circuit_results.csv")
print("========================================")