# Invention Disclosure Format (IDF)-B

**1. Title of the invention:**
Hardware-Agnostic Quantum Circuit Routing via Topological Graph Neural Networks (GNN)

**2. Field /Area of invention:**
Quantum Computing, Quantum Circuit Compilation, Graph Neural Networks (GNN)

**3. Prior Patents and Publications from literature:**
| Category | Title/Patent Number | Description | Publication Source | Year |
|----------|--------------------|-------------|--------------------|------|
| Quantum Routing | US 2021/0125134 A1 | Method for mapping quantum circuits using heuristic search and simulated annealing. | USPTO | 2021 |
| Reinforcement Learning | US 2023/0091234 A1 | Reinforcement learning agent for specific hardware topology routing. | USPTO | 2023 |
| SWAP Gate Minimization | US 2020/0342345 A1 | Distance-based SWAP insertion using tabular feature engineering for fixed hardware. | USPTO | 2020 |

**4. Summary and background of the invention (Address the gap / Novelty)**
**Summary:**
This invention provides a Graph Neural Network (GNN)-based quantum routing compiler that dynamically maps virtual qubits to physical qubits by modeling the hardware layout and circuit as topological graphs. It learns universal spatial routing laws, allowing it to inject the absolute minimum number of SWAP gates required to execute deep quantum circuits across dynamically changing hardware topologies.
**Background and Novelty:**
Traditional quantum compilers rely on rigid heuristic algorithms or tabular machine learning methods that memorize specific hardware board layouts (e.g., a fixed 5-qubit array). If the physical quantum chip breaks or scales up, traditional compilers completely fail and require total retraining. This invention addresses this gap by using a GNN to learn universal message-passing topological rules rather than memorizing a fixed board. The novelty lies in its hardware-agnostic generalizability: it can seamlessly adapt to any arbitrary or degrading physical quantum chip layout instantly without retraining, dramatically reducing SWAP overhead and preserving quantum coherence.

**5. Objective(s) of Invention**
The primary objective of this invention is to minimize the SWAP gate overhead in quantum circuit compilation, which is critical for reducing decoherence errors in Noisy Intermediate-Scale Quantum (NISQ) devices. Furthermore, the invention aims to provide a robust, hardware-agnostic routing framework that can dynamically adapt to arbitrary and faulty hardware topologies without requiring the costly retraining processes associated with traditional Reinforcement Learning or Random Forest compilers.

**6. Working principle of the invent (in brief)**
The system translates the physical quantum hardware (qubits and connections) and the upcoming logical gates into a singular graph representation. The GNN processes this graph via message passing, embedding spatial relationships, and outputs a prediction score for potential SWAP candidates. The system then dynamically injects the lowest-cost SWAP gate to route the qubits together.

**7. Description of the invention in detail**
**1. System Components:**
* **Topological Input Interface:** A module that converts arbitrary quantum hardware topologies (Coupling Maps) into normalized Adjacency Matrices.
* **GNN Processing Engine:** A multi-layer Graph Convolutional Network (GCNConv) that acts as the core predictive brain, equipped with global mean pooling to capture holistic board states.
* **Lookahead Subsystem:** A module that parses the upcoming sequence of quantum gates to feed future spatial requirements into the GNN node features.
**2. Operational Workflow:**
* **Graph Construction:** The current qubit mapping and upcoming gates are encoded as Node and Edge features on the physical hardware graph.
* **Message Passing:** The GNN engine processes the graph, allowing nodes to share spatial information about upcoming gate requirements.
* **Dynamic Routing:** The engine calculates the predicted future cost of all legal SWAP candidates and executes the one that optimally shortens the path for upcoming gates.
**3. Architecture Description:**
The architecture relies on a 3-layer GCNConv structure built on PyTorch Geometric. Node features include binary flags for whether a physical qubit is involved in the current or upcoming logical gates. By using graph message passing, the system intrinsically understands distance and bottlenecks without requiring handcrafted tabular features. 

**8. Experimental validation results:**
The invention was validated against a standard Greedy Baseline compiler across a rigorous stress test of 1,000 deep quantum circuits (20-40 gates). 
* **Baseline Total SWAPs:** 29,926 (Avg: 29.93 per circuit)
* **GNN Total SWAPs:** 23,301 (Avg: 23.30 per circuit)
The GNN achieved a massive ~22% reduction in SWAP gates compared to the baseline, definitively proving its efficacy at topological routing optimization.

**9. What aspect(s) of the invention need(s) protection?**
1. **Topological Modeling of Quantum States:** The specific method of translating physical quantum hardware and upcoming logical gates into a unified mathematical graph for neural network processing.
2. **Hardware-Agnostic GNN Routing:** The application of Graph Convolutional Networks (GCN) to predict quantum routing costs, allowing the compiler to function across dynamic, changing, or degrading physical chip topologies without retraining.
3. **Lookahead Node Feature Encoding:** The specific integration of future gate requirements directly into the node feature vectors of the physical hardware graph.

**10. Technology readiness level (TRL)**
TRL 6: Technology demonstrated in a relevant environment. (The GNN has been fully implemented, trained, and successfully integrated into a live web-based compiler prototype demonstrating real-time SWAP reduction).
