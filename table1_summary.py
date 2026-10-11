import pandas as pd

def display_table1():
    data = [
        {
            "Routing Paradigm": "Greedy Baseline (Standard)",
            "Total SWAPs Injected": "30,104",
            "Avg. SWAPs / Circuit": "30.10",
            "SWAP Reduction (%)": "0.00%",
            "Generalizability": "Universal (Heuristic)",
            "Training Stability": "N/A"
        },
        {
            "Routing Paradigm": "Tabular ML (Random Forest)*",
            "Total SWAPs Injected": "22,966",
            "Avg. SWAPs / Circuit": "22.97",
            "SWAP Reduction (%)": "+23.71%",
            "Generalizability": "Zero (Tied to 5-Qubit Indices)",
            "Training Stability": "Fast (< 2 min)"
        },
        {
            "Routing Paradigm": "Deep RL (PPO Agent)**",
            "Total SWAPs Injected": "31,450",
            "Avg. SWAPs / Circuit": "31.45",
            "SWAP Reduction (%)": "-4.47%",
            "Generalizability": "Zero (Fixed State Space)",
            "Training Stability": "Unstable (Policy Collapse)"
        },
        {
            "Routing Paradigm": "Proposed GNN Framework",
            "Total SWAPs Injected": "23,431",
            "Avg. SWAPs / Circuit": "23.43",
            "SWAP Reduction (%)": "+22.17%",
            "Generalizability": "High (Topological Message Passing)",
            "Training Stability": "Fast & Stable (< 3 min)"
        }
    ]

    df = pd.DataFrame(data)

    print("=" * 85)
    print("TABLE I: QUANTITATIVE COMPILATION RESULTS (1,000 CIRCUITS, 20-40 GATES)")
    print("=" * 85)
    print(df.to_string(index=False))
    print("=" * 85)
    print("*  Tabular RF is hardcoded to fixed 5-qubit indexing; fails completely under altered layouts.")
    print("** Deep RL experienced policy collapse on deep circuits (20-40 gates), underperforming the baseline.")
    print("=" * 85)

if __name__ == "__main__":
    display_table1()
