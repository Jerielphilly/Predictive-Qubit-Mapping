import random
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from router import route_circuit_baseline
from ml_router import ml_route_circuit as phase2_route
from gnn_router import ml_route_circuit as phase3_route


NUM_TESTS = 10
CIRCUITS_PER_TEST = 100
NUM_QUBITS = 5


coupling_map = CouplingMap([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4]
])


test_results = []
all_results = []


phase1_total = 0
phase2_total = 0
phase3_total = 0


phase1_wins = 0
phase2_wins = 0
phase3_wins = 0
same_tests = 0


print("========================================")
print("     PHASE 1 vs PHASE 2 vs PHASE 3")
print("========================================")
print()
print("Running 10 tests of 100 circuits each")
print("Total circuits: 1000")
print("Fresh random circuits: ENABLED")
print("Fixed seeds: DISABLED")
print()


for test_number in range(NUM_TESTS):

    test_phase1 = 0
    test_phase2 = 0
    test_phase3 = 0

    phase1_better = 0
    phase2_better = 0
    phase3_better = 0
    same = 0


    print("----------------------------------------")
    print(f"TEST {test_number + 1}")
    print("Generating 100 fresh random circuits...")
    print("----------------------------------------")


    for circuit_number in range(CIRCUITS_PER_TEST):

        circuit = QuantumCircuit(NUM_QUBITS)

        num_gates = random.randint(5, 15)


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


        _, swaps1 = route_circuit_baseline(
            circuit,
            coupling_map
        )


        _, swaps2 = phase2_route(
            circuit,
            coupling_map
        )


        _, swaps3 = phase3_route(
            circuit,
            coupling_map
        )


        test_phase1 += swaps1
        test_phase2 += swaps2
        test_phase3 += swaps3


        values = [
            swaps1,
            swaps2,
            swaps3
        ]


        minimum = min(values)


        winners = values.count(
            minimum
        )


        if winners > 1:

            same += 1

        elif swaps1 == minimum:

            phase1_better += 1

        elif swaps2 == minimum:

            phase2_better += 1

        elif swaps3 == minimum:

            phase3_better += 1


        all_results.append({
            "Test": test_number + 1,
            "Circuit": circuit_number + 1,
            "Phase 1": swaps1,
            "Phase 2": swaps2,
            "Phase 3": swaps3
        })


    phase1_avg = (
        test_phase1
        / CIRCUITS_PER_TEST
    )

    phase2_avg = (
        test_phase2
        / CIRCUITS_PER_TEST
    )

    phase3_avg = (
        test_phase3
        / CIRCUITS_PER_TEST
    )


    phase2_reduction = (
        (test_phase1 - test_phase2)
        / test_phase1
    ) * 100 if test_phase1 > 0 else 0


    phase3_reduction = (
        (test_phase1 - test_phase3)
        / test_phase1
    ) * 100 if test_phase1 > 0 else 0


    test_totals = [
        test_phase1,
        test_phase2,
        test_phase3
    ]


    minimum_total = min(
        test_totals
    )


    if test_totals.count(minimum_total) > 1:

        winner = "Same"
        same_tests += 1

    elif test_phase1 == minimum_total:

        winner = "Phase 1"
        phase1_wins += 1

    elif test_phase2 == minimum_total:

        winner = "Phase 2"
        phase2_wins += 1

    else:

        winner = "Phase 3"
        phase3_wins += 1


    test_results.append({
        "Test": test_number + 1,
        "Circuits": CIRCUITS_PER_TEST,

        "Phase 1 Total": test_phase1,
        "Phase 2 Total": test_phase2,
        "Phase 3 Total": test_phase3,

        "Phase 1 Average": round(
            phase1_avg,
            2
        ),

        "Phase 2 Average": round(
            phase2_avg,
            2
        ),

        "Phase 3 Average": round(
            phase3_avg,
            2
        ),

        "Phase 2 Reduction (%)": round(
            phase2_reduction,
            2
        ),

        "Phase 3 Reduction (%)": round(
            phase3_reduction,
            2
        ),

        "Phase 1 Better": phase1_better,
        "Phase 2 Better": phase2_better,
        "Phase 3 Better": phase3_better,
        "Same": same,

        "Winner": winner
    })


    phase1_total += test_phase1
    phase2_total += test_phase2
    phase3_total += test_phase3


    print()
    print(
        f"Phase 1 total : {test_phase1}"
    )

    print(
        f"Phase 2 total : {test_phase2}"
    )

    print(
        f"Phase 3 total : {test_phase3}"
    )

    print()

    print(
        f"Phase 2 reduction : "
        f"{phase2_reduction:.2f}%"
    )

    print(
        f"Phase 3 reduction : "
        f"{phase3_reduction:.2f}%"
    )

    print(
        f"Winner : {winner}"
    )

    print()


total_circuits = (
    NUM_TESTS
    * CIRCUITS_PER_TEST
)


phase1_average = (
    phase1_total
    / total_circuits
)

phase2_average = (
    phase2_total
    / total_circuits
)

phase3_average = (
    phase3_total
    / total_circuits
)


phase2_overall_reduction = (
    (phase1_total - phase2_total)
    / phase1_total
) * 100 if phase1_total > 0 else 0


phase3_overall_reduction = (
    (phase1_total - phase3_total)
    / phase1_total
) * 100 if phase1_total > 0 else 0


df_tests = pd.DataFrame(
    test_results
)

df_all = pd.DataFrame(
    all_results
)


df_tests.to_csv(
    "phase3_test_results.csv",
    index=False
)

df_all.to_csv(
    "phase3_all_circuit_results.csv",
    index=False
)


print()
print("========================================")
print("           FINAL TEST TABLE")
print("========================================")
print()

print(
    df_tests.to_string(
        index=False
    )
)


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
    f"Phase 1 total SWAPs   : "
    f"{phase1_total}"
)

print(
    f"Phase 2 total SWAPs   : "
    f"{phase2_total}"
)

print(
    f"Phase 3 total SWAPs   : "
    f"{phase3_total}"
)

print()

print(
    f"Phase 1 average SWAPs : "
    f"{phase1_average:.2f}"
)

print(
    f"Phase 2 average SWAPs : "
    f"{phase2_average:.2f}"
)

print(
    f"Phase 3 average SWAPs : "
    f"{phase3_average:.2f}"
)

print()

print(
    f"Phase 1 won tests     : "
    f"{phase1_wins}"
)

print(
    f"Phase 2 won tests     : "
    f"{phase2_wins}"
)

print(
    f"Phase 3 won tests     : "
    f"{phase3_wins}"
)

print(
    f"Same tests            : "
    f"{same_tests}"
)

print()

print(
    f"Phase 2 reduction     : "
    f"{phase2_overall_reduction:.2f}%"
)

print(
    f"Phase 3 reduction     : "
    f"{phase3_overall_reduction:.2f}%"
)

print()

print("========================================")
print("Files saved:")
print("phase3_test_results.csv")
print("phase3_all_circuit_results.csv")
print("========================================")
print()
print("Fresh random circuits were used.")
print("Fixed seeds: DISABLED")
print("========================================")