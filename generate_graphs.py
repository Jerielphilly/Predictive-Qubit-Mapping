import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/benchmark_all_results.csv")

base_avg = df["Baseline"].mean()
rf_avg = df["Random Forest"].mean()
gnn_avg = df["GNN"].mean()

print("========================================")
print("          FINAL ANALYSIS")
print("========================================")
print(f"Baseline avg SWAPs     : {base_avg:.2f}")
print(f"Random Forest avg      : {rf_avg:.2f}")
print(f"GNN avg                : {gnn_avg:.2f}")
print("========================================")

# Graph 1: Line plot of a single test
df_subset = df[df["Test"] == 1]
plt.figure(figsize=(12, 6))
plt.plot(df_subset["Circuit"], df_subset["Baseline"], label="Baseline (Greedy)", alpha=0.5)
plt.plot(df_subset["Circuit"], df_subset["Random Forest"], label="Random Forest", alpha=0.8)
plt.plot(df_subset["Circuit"], df_subset["GNN"], label="Graph Neural Network", alpha=0.9, linewidth=2)
plt.xlabel("Circuit Number")
plt.ylabel("Number of SWAP Gates")
plt.title("SWAP Count per Circuit (First 100 Circuits)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("results/graph1_benchmark_line.png", dpi=300, bbox_inches="tight")
plt.close()

# Graph 2: Bar chart
plt.figure(figsize=(10, 6))
bars = plt.bar(["Baseline", "Random Forest", "GNN"], 
               [base_avg, rf_avg, gnn_avg], 
               color=['#1f77b4', '#ff7f0e', '#d62728'])
plt.ylabel("Average SWAP Gates")
plt.title("Average SWAPs Injected (Deep 20-40 Gate Circuits)")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, round(yval, 2), ha='center', va='bottom')
plt.savefig("results/graph2_average_swaps.png", dpi=300, bbox_inches="tight")
plt.close()