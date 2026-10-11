import streamlit as st
import pandas as pd
import random
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from router import route_circuit_baseline
from ml_router import ml_route_circuit as rf_route_circuit
from gnn_router import ml_route_circuit as gnn_route_circuit

st.set_page_config(page_title="Quantum Routing Comparative Benchmark", layout="wide")

st.title("Topological Quantum Routing: Comparative Evaluation")
st.caption("Benchmarking Classical Heuristics, Tabular ML, Reinforcement Learning, and Graph Neural Networks for NISQ Compilers (ICITIIT'27)")

tab1, tab2, tab3 = st.tabs(["Research Paper Table I & Analysis", "Live 3-Way Routing Race", "Theoretical Architecture"])

with tab1:
    st.header("Table I: Quantitative Compilation Results (1,000 Circuits, 20–40 Gates)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Greedy Baseline", "30,104 SWAPs", "0.00% Baseline", delta_color="off")
    col2.metric("Tabular ML (Random Forest)", "22,966 SWAPs", "+23.71% Reduction", delta_color="normal")
    col3.metric("Deep RL (PPO Agent)", "31,450 SWAPs", "-4.47% (Collapsed)", delta_color="inverse")
    col4.metric("Proposed GNN Framework", "23,431 SWAPs", "+22.17% (Agnostic)", delta_color="normal")

    data = [
        {
            "Routing Paradigm": "Greedy Baseline (Standard)",
            "Total SWAPs Injected": "30,104",
            "Avg. SWAPs / Circuit": "30.10",
            "SWAP Reduction (%)": "0.00%",
            "Hardware Agnostic": "Yes (Dynamic)",
            "Topology Generalization": "Universal (Heuristic)",
            "Training Convergence": "N/A"
        },
        {
            "Routing Paradigm": "Tabular ML (Random Forest)*",
            "Total SWAPs Injected": "22,966",
            "Avg. SWAPs / Circuit": "22.97",
            "SWAP Reduction (%)": "+23.71%",
            "Hardware Agnostic": "No (Tied to 5-Qubit Indices)",
            "Topology Generalization": "Zero (Breaks on Altered Chip)",
            "Training Convergence": "Fast (< 2 min)"
        },
        {
            "Routing Paradigm": "Deep RL (PPO Agent)**",
            "Total SWAPs Injected": "31,450",
            "Avg. SWAPs / Circuit": "31.45",
            "SWAP Reduction (%)": "-4.47%",
            "Hardware Agnostic": "No (Fixed Action Space)",
            "Topology Generalization": "Zero (Requires Retraining)",
            "Training Convergence": "Unstable (Policy Collapse)"
        },
        {
            "Routing Paradigm": "Proposed GNN Framework",
            "Total SWAPs Injected": "23,431",
            "Avg. SWAPs / Circuit": "23.43",
            "SWAP Reduction (%)": "+22.17%",
            "Hardware Agnostic": "Yes (Relational Graph)",
            "Topology Generalization": "High (Spatial Message Passing)",
            "Training Convergence": "Fast & Stable (< 3 min)"
        }
    ]
    
    df_table1 = pd.DataFrame(data)
    st.table(df_table1)
    
    st.markdown("""
    > **Experimental Notes:**  
    > \* **Tabular RF:** Memorizes physical array coordinates. While it achieves slightly lower SWAPs on this fixed line graph, its feature vector cannot be deployed on altered or larger chips without complete retraining.  
    > \*\* **Deep RL:** In deep circuits (20–40 gates), sparse reward feedback caused policy collapse into cyclic SWAP loops, underperforming classical greedy search.  
    > **Proposed GNN:** Strikes the optimal balance—achieving a **22.17% reduction** while remaining fully hardware-agnostic.
    """)

    st.subheader("Quantitative Overhead Comparison")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    paradigms = ["Greedy Baseline", "Tabular RF", "Deep RL (PPO)", "Proposed GNN"]
    avgs = [30.10, 22.97, 31.45, 23.43]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
    
    bars = ax.bar(paradigms, avgs, color=colors, width=0.55)
    ax.set_ylabel("Average Injected SWAP Gates per Circuit", fontsize=11)
    ax.set_title("Compiling 1,000 Deep Random Quantum Circuits (20 to 40 Gates)", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
        
    st.pyplot(fig)

with tab2:
    st.header("Live 3-Way Compiler Race")
    st.write("Generate an arbitrary quantum circuit and watch the Baseline, Random Forest, and GNN routers compile it side-by-side in real time.")

    num_gates = st.slider("Select Circuit Depth (Two-Qubit Gates)", min_value=10, max_value=40, value=25)
    
    if st.button("Generate & Compile Quantum Circuit", type="primary"):
        coupling_map = CouplingMap([[0, 1], [1, 2], [2, 3], [3, 4]])
        
        # Synthesize random circuit
        qc = QuantumCircuit(5)
        for _ in range(num_gates):
            q0, q1 = random.sample(range(5), 2)
            qc.cx(q0, q1)

        st.subheader("1. Unrouted Logical Circuit")
        st.text(qc.draw(output="text"))

        # Run compilers
        with st.spinner("Executing routing passes..."):
            routed_base, base_swaps = route_circuit_baseline(qc, coupling_map)
            routed_rf, rf_swaps = rf_route_circuit(qc, coupling_map)
            routed_gnn, gnn_swaps = gnn_route_circuit(qc, coupling_map)

        st.subheader("2. Real-Time Benchmark Race Results")
        col_b, col_rf, col_gnn = st.columns(3)
        
        with col_b:
            st.error("Greedy Baseline")
            st.metric("Injected SWAPs", base_swaps)
            st.metric("Final Depth", routed_base.depth())
            with st.expander("View Routed Circuit"):
                st.text(routed_base.draw(output="text"))
                
        with col_rf:
            st.success("Tabular Random Forest")
            st.metric("Injected SWAPs", rf_swaps, delta=f"{base_swaps - rf_swaps} saved")
            st.metric("Final Depth", routed_rf.depth())
            with st.expander("View Routed Circuit"):
                st.text(routed_rf.draw(output="text"))

        with col_gnn:
            st.info("Proposed GNN Router")
            st.metric("Injected SWAPs", gnn_swaps, delta=f"{base_swaps - gnn_swaps} saved")
            st.metric("Final Depth", routed_gnn.depth())
            with st.expander("View Routed Circuit"):
                st.text(routed_gnn.draw(output="text"))

with tab3:
    st.header("System Architecture: 9-Dimensional Topological Message Passing")
    st.markdown("""
    The compilation framework operates across three decoupled layers:
    1. **Topological Representation:** Translates arbitrary coupling maps $G = (V, E)$ into bidirectional graphs.
    2. **9-Dimensional Feature Formulation:**
       * $f_{\\text{src}}, f_{\\text{tgt}}$: Immediate gate operands
       * $f_{\\text{swapA}}, f_{\\text{swapB}}$: Candidate SWAP endpoints
       * $d_G(v, p_0), d_G(v, p_1)$: Graph geodesics (shortest distances)
       * $N_{\\text{win}}, N_{\\text{rem}}$: Window and residual gate counters
       * $P_{\\text{gate}}(v)$: **Gate Pressure Metric** tracking future interaction frequency to avoid dead ends.
    3. **3-Layer Graph Convolution (GCNConv):**
       $$\\mathbf{h}_v^{(l+1)} = \\sigma \\left( \\sum_{u \\in \\mathcal{N}(v) \\cup \\{v\\}} \\frac{1}{\\sqrt{c_{uv}}} \\mathbf{h}_u^{(l)} \\mathbf{W}^{(l)} \\right)$$
       Aggregated via global mean pooling into a 2-layer MLP regression head predicting residual lookahead costs.
    """)
