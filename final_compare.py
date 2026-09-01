import random
import pandas as pd

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from router import route_circuit_baseline
from ml_router import ml_route_circuit as phase2_router
from gnn_router import ml_route_circuit as phase3_router


NUM_TESTS = 10
CIRCUITS_PER_TEST = 100

SEEDS = [
    101,
    202,
    303,
    404,
    505,
    606,
    707,
    808,
    909,
    1010
]


coupling_map = CouplingMap([
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4]
])


test_results = []
all_results = []


overall_phase1 = 0
overall_phase2 = 0
overall_phase3 = 0


phase1_wins = 0
phase2_wins = 0
phase3_wins = 0
same_tests = 0


print("========================================")
print("     PHASE 1 vs PHASE 2 vs PHASE 3")
print("========================================")
print()
print("Running 10 independent tests")
print("100 random circuits per test")
print("Total circuits: 1000")
print()


for test_number in range(NUM_TESTS):

    seed = SEEDS[test_number]

    random.seed(seed)

    phase1_total = 0
    phase2_total = 0
    phase3_total = 0

    phase1_better = 0
    phase2_better = 0
    phase3_better = 0
    same_result = 0


    print("----------------------------------------")
    print(f"TEST {test_number + 1}")
    print(f"Seed: {seed}")
    print("Running 100 circuits...")
    print("----------------------------------------")


    for circuit_number in range(
        CIRCUITS_PER_TEST
    ):

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


        _, phase2_swaps = phase2_router(
            circuit,
            coupling_map
        )


        _, phase3_swaps = phase3_router(
            circuit,
            coupling_map
        )


        phase1_total += phase1_swaps
        phase2_total += phase2_swaps
        phase3_total += phase3_swaps


        minimum = min(
            phase1_swaps,
            phase2_swaps,
            phase3_swaps
        )


        winners = 0

        if phase1_swaps == minimum:
            winners += 1

        if phase2_swaps == minimum:
            winners += 1

        if phase3_swaps == minimum:
            winners += 1


        if winners == 1:

            if phase1_swaps == minimum:
                phase1_better += 1

            elif phase2_swaps == minimum:
                phase2_better += 1

            else:
                phase3_better += 1

        else:

            same_result += 1


        all_results.append({

            "Test": test_number + 1,

            "Seed": seed,

            "Circuit": circuit_number + 1,

            "Phase 1": phase1_swaps,

            "Phase 2": phase2_swaps,

            "Phase 3": phase3_swaps

        })


    phase1_average = (
        phase1_total
        / CIRCUITS_PER_TEST
    )


    phase2_average = (
        phase2_total
        / CIRCUITS_PER_TEST
    )


    phase3_average = (
        phase3_total
        / CIRCUITS_PER_TEST
    )


    reduction_phase2 = (

        (
            phase1_total
            - phase2_total
        )
        / phase1_total
        * 100

    ) if phase1_total > 0 else 0


    reduction_phase3 = (

        (
            phase1_total
            - phase3_total
        )
        / phase1_total
        * 100

    ) if phase1_total > 0 else 0


    totals = {
        "Phase 1": phase1_total,
        "Phase 2": phase2_total,
        "Phase 3": phase3_total
    }


    best_total = min(
        totals.values()
    )


    winning_phases = [

        phase

        for phase, total
        in totals.items()

        if total == best_total

    ]


    if len(winning_phases) == 1:

        winner = winning_phases[0]

        if winner == "Phase 1":
            phase1_wins += 1

        elif winner == "Phase 2":
            phase2_wins += 1

        else:
            phase3_wins += 1

    else:

        winner = "Same"

        same_tests += 1


    test_results.append({

        "Test": test_number + 1,

        "Seed": seed,

        "Circuits": CIRCUITS_PER_TEST,

        "Phase 1 Total": phase1_total,

        "Phase 2 Total": phase2_total,

        "Phase 3 Total": phase3_total,

        "Phase 1 Average": round(
            phase1_average,
            2
        ),

        "Phase 2 Average": round(
            phase2_average,
            2
        ),

        "Phase 3 Average": round(
            phase3_average,
            2
        ),

        "Phase 2 Reduction (%)": round(
            reduction_phase2,
            2
        ),

        "Phase 3 Reduction (%)": round(
            reduction_phase3,
            2
        ),

        "Phase 1 Better": phase1_better,

        "Phase 2 Better": phase2_better,

        "Phase 3 Better": phase3_better,

        "Same": same_result,

        "Winner": winner

    })


    overall_phase1 += phase1_total
    overall_phase2 += phase2_total
    overall_phase3 += phase3_total


    print()
    print(
        f"Phase 1 total : {phase1_total}"
    )

    print(
        f"Phase 2 total : {phase2_total}"
    )

    print(
        f"Phase 3 total : {phase3_total}"
    )

    print()

    print(
        f"Phase 3 reduction vs Phase 1 : "
        f"{reduction_phase3:.2f}%"
    )

    print(
        f"Winner : {winner}"
    )

    print()


total_circuits = (
    NUM_TESTS
    * CIRCUITS_PER_TEST
)


overall_reduction_phase2 = (

    (
        overall_phase1
        - overall_phase2
    )
    / overall_phase1
    * 100

) if overall_phase1 > 0 else 0


overall_reduction_phase3 = (

    (
        overall_phase1
        - overall_phase3
    )
    / overall_phase1
    * 100

) if overall_phase1 > 0 else 0


overall_average_phase1 = (
    overall_phase1
    / total_circuits
)


overall_average_phase2 = (
    overall_phase2
    / total_circuits
)


overall_average_phase3 = (
    overall_phase3
    / total_circuits
)


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
print("          TEST-BY-TEST RESULTS")
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
    "Phase 1 total SWAPs   :",
    overall_phase1
)

print(
    "Phase 2 total SWAPs   :",
    overall_phase2
)

print(
    "Phase 3 total SWAPs   :",
    overall_phase3
)

print()

print(
    "Phase 1 average SWAPs :",
    f"{overall_average_phase1:.2f}"
)

print(
    "Phase 2 average SWAPs :",
    f"{overall_average_phase2:.2f}"
)

print(
    "Phase 3 average SWAPs :",
    f"{overall_average_phase3:.2f}"
)

print()

print(
    "Phase 1 won tests     :",
    phase1_wins
)

print(
    "Phase 2 won tests     :",
    phase2_wins
)

print(
    "Phase 3 won tests     :",
    phase3_wins
)

print(
    "Same tests            :",
    same_tests
)

print()

print(
    "Phase 2 reduction     :",
    f"{overall_reduction_phase2:.2f}%"
)

print(
    "Phase 3 reduction     :",
    f"{overall_reduction_phase3:.2f}%"
)

print()
print("========================================")
print("Files saved:")
print("phase3_test_results.csv")
print("phase3_all_circuit_results.csv")
print("========================================")