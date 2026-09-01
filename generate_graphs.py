import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("comparison_results.csv")


phase1_total = df["Phase 1"].sum()
phase2_total = df["Phase 2"].sum()

phase1_average = df["Phase 1"].mean()
phase2_average = df["Phase 2"].mean()

phase2_better = (df["Phase 2"] < df["Phase 1"]).sum()
phase1_better = (df["Phase 1"] < df["Phase 2"]).sum()
same_result = (df["Phase 1"] == df["Phase 2"]).sum()

reduction = ((phase1_total - phase2_total) / phase1_total) * 100


print("========================================")
print("          FINAL ANALYSIS")
print("========================================")

print(f"Phase 1 total SWAPs   : {phase1_total}")
print(f"Phase 2 total SWAPs   : {phase2_total}")

print(f"Phase 1 average SWAPs : {phase1_average:.2f}")
print(f"Phase 2 average SWAPs : {phase2_average:.2f}")

print(f"Phase 2 better        : {phase2_better}")
print(f"Phase 1 better        : {phase1_better}")
print(f"Same result           : {same_result}")

print(f"SWAP reduction        : {reduction:.2f}%")
print("========================================")


# Graph 1
plt.figure(figsize=(12, 6))

plt.plot(
    df["Circuit"],
    df["Phase 1"],
    label="Phase 1"
)

plt.plot(
    df["Circuit"],
    df["Phase 2"],
    label="Phase 2"
)

plt.xlabel("Circuit Number")
plt.ylabel("Number of SWAP Gates")
plt.title("Phase 1 vs Phase 2 SWAP Count")
plt.legend()
plt.grid(True)

plt.savefig(
    "graph1_phase1_vs_phase2.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Graph 2
plt.figure(figsize=(7, 5))

plt.bar(
    ["Phase 1", "Phase 2"],
    [phase1_average, phase2_average]
)

plt.ylabel("Average SWAP Gates")
plt.title("Average SWAP Count Comparison")

plt.savefig(
    "graph2_average_swaps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Graph 3
plt.figure(figsize=(7, 5))

plt.bar(
    ["Phase 2 Better", "Phase 1 Better", "Same"],
    [phase2_better, phase1_better, same_result]
)

plt.ylabel("Number of Circuits")
plt.title("Circuit Performance Comparison")

plt.savefig(
    "graph3_circuit_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print()
print("Graphs generated successfully!")
print("graph1_phase1_vs_phase2.png")
print("graph2_average_swaps.png")
print("graph3_circuit_performance.png")