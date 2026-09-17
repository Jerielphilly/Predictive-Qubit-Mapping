# Predictive Qubit Mapping: Hardware-Agnostic Quantum Routing via Topological GNNs

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Qiskit](https://img.shields.io/badge/Qiskit-%236929C4.svg?style=for-the-badge&logo=Qiskit&logoColor=white)

This repository contains the code for a quantum circuit routing compiler we built. It uses a Graph Neural Network (GNN) to figure out the best places to insert SWAP gates so that quantum circuits can actually run on physical hardware. 

The main problem with Noisy Intermediate-Scale Quantum (NISQ) devices is that adding too many SWAP gates causes decoherence. While traditional compilers usually rely on fixed search heuristics or machine learning models that just memorize one specific chip layout, we wanted to build something completely hardware-agnostic. Our GNN actually learns the topological rules of how qubits interact, meaning it can dynamically route circuits over any physical coupling map.

## Key Results
To prove that this works, we ran a massive stress test generating 1,000 highly entangled random quantum circuits (each packed with 20 to 40 logical gates) and routed them onto a 5-qubit array. 

*   **Greedy Baseline Compiler:** 30,104 total SWAPs injected
*   **Our GNN Compiler:** 23,431 total SWAPs injected

Our model achieved a **~22.1% reduction** in SWAP overhead compared to the standard baseline. This directly translates to shorter physical circuits and fewer decoherence errors.

<p align="center">
  <img src="results/graph2_average_swaps.png" width="45%" />
  <img src="results/graph1_benchmark_line.png" width="45%" />
</p>

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jerielphilly/Predictive-Qubit-Mapping.git
   cd Predictive-Qubit-Mapping
   ```

2. **Set up a virtual environment (Recommended):**
   *Windows:*
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
   *macOS/Linux:*
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You'll want to make sure PyTorch and PyTorch Geometric are correctly configured for your specific OS and CUDA version).*

---

## Usage

### 1. Interactive Streamlit Dashboard (Live Demo)
We built a web dashboard to make it easy to visualize how the routing works in real-time. It dynamically generates random quantum circuits and runs a side-by-side compilation race between the Baseline method and our GNN.

```bash
streamlit run app.py
```
Just open `http://localhost:8501` in your browser. From there you can read up on the methodology, run the 1000-circuit benchmark yourself, and watch the live routing demo physically draw the compiled Qiskit circuits on the screen.

### 2. Run the Benchmark Locally
If you just want to run the math headless and generate the performance graphs:

```bash
# Run the benchmark (saves to results/benchmark_all_results.csv)
python compare.py

# Generate the visual graphs (saves PNGs to the results folder)
python generate_graphs.py
```

### 3. Retrain the GNN Model
If you want to tweak the node features or train the model from scratch:

```bash
# Generate 5,000 random circuits and encode them into PyTorch Geometric graphs
python gnn_dataset.py

# Train the 3-layer GCNConv model (Saves the weights to models/gnn_routing_model.pt)
python train_gnn.py
```

---

## How it Works (System Architecture)

```mermaid
graph TD
    %% Input Layer
    subgraph Input Processing
        A[Physical Qubit Coupling Map] --> C[Graph Constructor]
        B[Quantum Circuit Gates] --> C
    end

    %% GNN Engine
    subgraph Graph Neural Network Engine
        C --> D[Node Feature Encoder]
        D --> E[Lookahead Subsystem]
        
        E --> F[GCNConv Layer 1]
        F --> G[GCNConv Layer 2]
        G --> H[GCNConv Layer 3]
        
        H --> I[Global Mean Pooling]
    end

    %% Output Layer
    subgraph Routing Execution
        I --> J[Candidate SWAP Evaluator]
        J --> K[Cost Prediction Scoring]
        K --> L[Optimal SWAP Injection]
    end

    %% Styling
    classDef input fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000;
    classDef engine fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000;
    classDef output fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000;
    
    class A,B,C input;
    class D,E,F,G,H,I engine;
    class J,K,L output;
```

The compiler is broken down into three main steps:
1. **Graph Construction:** We take the physical quantum chip layout and the upcoming logical gates and encode them together into a PyTorch Geometric graph.
2. **Feature Encoding & Lookahead:** Every qubit (node) gets a 9-dimensional feature vector. This tracks things like shortest-path distances and a forward-looking "gate-pressure" score to see how heavily a qubit will be used in the future.
3. **Routing Inference:** We pass this graph through a 3-layer GCNConv network. After global mean pooling, a regression head predicts exactly how much routing cost a potential SWAP will incur in the future. The system then dynamically picks the SWAP with the lowest predicted cost and moves on to the next gate.

---

## Repository Structure

```text
├── data/                       # Datasets and generated graph data
├── models/
│   └── gnn_routing_model.pt    # Pre-trained GNN model weights
├── results/                    # CSV outputs and generated PNG graphs
├── app.py                      # Streamlit dashboard application
├── architecture_diagram.md     # Mermaid.js system block diagram
├── compare.py                  # Benchmark script (GNN vs Greedy)
├── generate_graphs.py          # Data visualization script
├── gnn_dataset.py              # PyTorch Geometric dataset generator
├── gnn_router.py               # Core GNN routing inference logic
├── router.py                   # Baseline shortest-path heuristic router
├── train_gnn.py                # Model training script
└── requirements.txt            # Python dependencies
```
