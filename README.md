# A Topological Graph Neural Network Framework for Hardware-Agnostic Quantum Circuit Routing on NISQ Architectures

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-%236929C4.svg?style=for-the-badge&logo=Qiskit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)

> 📄 **Manuscript:** [`research_paper.pdf`](research_paper.pdf)  
> 🔬 **Branch Overview (`ieee-paper-benchmarks`):** This branch serves as the complete reproducibility archive for our multi-paradigm research paper, containing all four routing implementations (**Greedy Heuristic**, **Tabular Random Forest**, **Deep RL PPO**, and **Topological GNN**), evaluation datasets, multi-topology transfer scripts, and ablation experiments.  
> For the standalone GNN quantum compiler package, see the [`main`](https://github.com/Jerielphilly/Predictive-Qubit-Mapping/tree/main) branch.

---

## Abstract Summary
Executing multi-qubit quantum circuits on Noisy Intermediate-Scale Quantum (NISQ) processors requires routing non-adjacent logical two-qubit operations onto constrained physical coupling graphs via dynamic SWAP insertion. We propose a **Topological Graph Neural Network (GNN) Routing Framework** that models physical chip connectivity and pending gate operations via a 9-dimensional node feature representation and a 3-layer Graph Convolutional Network (`GCNConv`) with global mean pooling.

We systematically benchmark the proposed GNN framework against three alternative compilation paradigms under identical initial layout conditions ($\pi_0 = \text{Id}$):
1. **Greedy Baseline (`router.py`):** Classical sequential shortest-path routing.
2. **Tabular Supervised ML (`ml_router.py`):** Random Forest regressor trained on static physical index features.
3. **Model-Free Deep RL (`rl_env.py`, `train_rl.py`):** Proximal Policy Optimization (PPO) agent trained over 50M steps.
4. **Proposed Topological GNN (`gnn_router.py`):** 3-layer GCN trained on 112,136 graph states (from 140,170 total states extracted across 5,000 circuits).

---

## Experimental Results

### 1. Primary Benchmark (1,000 Deep Circuits, 21–40 Entangling Gates)
Evaluated on a 5-qubit linear transmon topology ($V=\{0,1,2,3,4\}$):

| Routing Paradigm | Total SWAPs Injected | Avg. SWAPs / Circuit | SWAP Reduction (%) | Compile Latency |
| :--- | :---: | :---: | :---: | :---: |
| **Greedy Baseline (Standard)** | 30,104 | 30.10 | 0.00% | 0.7 ms |
| **Tabular ML (Random Forest)** | 22,966 | 22.97 | 23.71% | 12.4 ms |
| **Deep RL (PPO Agent)** | 31,450 | 31.45 | -4.47% | 18.2 ms |
| **Proposed GNN Framework** | **23,431** | **23.43** | **22.17%** | **35.5 ms** |

* **Statistical Significance:** Two-tailed paired Student's t-test confirms $t = 28.42, p < 10^{-50}$ (95% CI: $[6.21, 7.13]$ SWAPs saved per circuit).
* **Instance-Level Non-Inferiority:** The GNN outperforms or matches the greedy baseline on **901 out of 1,000 circuits** (746 strict wins, 155 ties; **90.1%** non-inferiority rate).

<p align="center">
  <img src="results/graph2_average_swaps.png" width="45%" />
  <img src="results/graph1_benchmark_line.png" width="45%" />
</p>

---

### 2. Zero-Shot Cross-Topology Transfer (30–40 Gate Deep Circuits)
Trained exclusively on the 5-qubit linear path, the GNN is evaluated **zero-shot (without retraining)** across unseen branching, cyclic, and 7-qubit processor topologies:

| Topology / Architecture | Qubit Count | Baseline Avg | Tabular RF Avg | GNN Avg | GNN vs. Baseline | GNN vs. RF |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Path (Trained)** | 5 | 34.82 | 26.56 | **25.38** | **+27.1%** | **+4.4%** |
| **T-Junction (Zero-Shot)** | 5 | 27.66 | 23.72 | **23.14** | **+16.34%** | **+2.4%** |
| **Cyclic Ring (Zero-Shot)** | 5 | 17.32 | 15.76 | **13.12** | **+24.25%** | **+16.8%** |
| **IBM Falcon Subgraph (Zero-Shot)** | 7 | 49.06 | 41.10 | **41.40** | **+15.61%** | Comparable |

---

### 3. Inference-Time Sensitivity & Horizon Ablation (50 Paired Circuits, 25 Gates)

| Configuration / Variant | Evaluated Component | Avg. SWAPs ($\pm$ std) | Paired $\Delta$ (Mean $\pm$ SE) | Relative Degradation |
| :--- | :--- | :---: | :---: | :---: |
| **Full Proposed Framework** | Full 9D Model ($W=6$) | **19.80 $\pm$ 2.41** | — | **Baseline (0.0%)** |
| **Ablation 1: No Gate Pressure** | Zero-Masked Gate Pressure | 21.76 $\pm$ 2.85 | +1.96 $\pm$ 0.31 | **+9.9% (Worse)** |
| **Ablation 2: Truncated Horizon** | Lookahead Window $W=2$ | 21.14 $\pm$ 2.62 | +1.34 $\pm$ 0.28 | **+6.8% (Worse)** |
| **Ablation 3: Extended Horizon** | Lookahead Window $W=25$ | 22.55 $\pm$ 3.10 | +2.75 $\pm$ 0.39 | **+13.9% (Worse)** |

---

## Reproducing the Paper Benchmarks

### 1. Environment Setup
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Print Table I Summary
```powershell
python table1_summary.py
```

### 3. Run the 4-Paradigm Interactive Dashboard
```powershell
python -m streamlit run app.py
```

### 4. Run Zero-Shot Cross-Topology Benchmarks (Table III)
```powershell
# GNN vs. Baseline across 4 topologies
python benchmark_multitopology.py

# Deep 30-40 gate stress test comparing Greedy vs. Tabular RF vs. GNN
python test_rf_vs_gnn_deep.py
```

---

## Repository Structure (`ieee-paper-benchmarks`)

```text
├── research_paper.pdf                 # Final 6-page IEEE Conference Manuscript
├── app.py                             # 4-Paradigm Streamlit Research Dashboard
├── table1_summary.py                  # Reproduces Table I quantitative summary
├── benchmark_multitopology.py         # Zero-shot multi-topology transfer script
├── test_rf_vs_gnn_deep.py             # Deep 30-40 gate GNN vs. RF cross-topology test
├── router.py                          # Paradigm 1: Classical Greedy Baseline Router
├── ml_router.py                       # Paradigm 2: Tabular Random Forest Router
├── rl_env.py                          # Paradigm 3: Deep RL (PPO) Gymnasium Environment
├── gnn_router.py                      # Paradigm 4: Proposed Topological GNN Router
├── gnn_dataset.py                     # 9D graph feature dataset generator
├── dataset_generator.py               # Tabular lookahead dataset generator
├── train_gnn.py                       # 3-layer GCNConv training script
├── train_model.py                     # Tabular Random Forest training script
├── train_rl.py                        # PPO Reinforcement Learning training script
├── gnn_routing_model.pt               # Trained PyTorch Geometric GNN weights
├── routing_model.pkl                  # Trained Scikit-Learn Random Forest weights
└── results/
    ├── benchmark_1000_circuits.csv    # 1,000-circuit evaluation logs (Baseline, RF, GNN)
    ├── multitopology_results.csv      # Cross-topology benchmark outputs
    ├── graph1_benchmark_line.png      # 1,000-circuit SWAP count trajectory plot
    └── graph2_average_swaps.png       # Average SWAP overhead bar chart
```
