import gymnasium as gym
import numpy as np
from gymnasium import spaces
import networkx as nx
import random

class QuantumRoutingEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    The agent must route a sequence of 2-qubit gates on a 5-qubit linear array.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super(QuantumRoutingEnv, self).__init__()

        self.num_qubits = 5
        self.lookahead = 5
        self.max_steps = 200

        # Actions: 4 possible physical SWAPs on a line graph [0-1, 1-2, 2-3, 3-4]
        self.action_space = spaces.Discrete(4)

        # Observation:
        # [0-4]: which virtual qubit is at physical node i
        # [5-14]: the next 5 gates (v0, v1). Padded with -1 if circuit is ending.
        self.observation_space = spaces.Box(
            low=-1, high=self.num_qubits - 1, shape=(5 + self.lookahead * 2,), dtype=np.int32
        )

        self.graph = nx.path_graph(self.num_qubits)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate a random circuit of 20-30 gates
        num_gates = self.np_random.integers(20, 30)
        self.gates = []
        for _ in range(num_gates):
            q0, q1 = self.np_random.choice(self.num_qubits, 2, replace=False)
            self.gates.append((q0, q1))

        self.current_step = 0
        self.gate_index = 0
        
        # physical_to_virtual mapping: index is physical node, value is virtual qubit
        self.layout = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        
        # Auto-execute any gates that happen to already be adjacent
        self._auto_execute()

        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.full(5 + self.lookahead * 2, -1, dtype=np.int32)
        
        # 1. Current Layout
        obs[0:5] = self.layout
        
        # 2. Upcoming Gates
        remaining_gates = self.gates[self.gate_index : self.gate_index + self.lookahead]
        for i, (q0, q1) in enumerate(remaining_gates):
            obs[5 + i * 2] = q0
            obs[6 + i * 2] = q1
            
        return obs

    def _auto_execute(self):
        """Executes the current target gate if its qubits are adjacent in the layout."""
        executed_count = 0
        while self.gate_index < len(self.gates):
            q0, q1 = self.gates[self.gate_index]
            
            # Find physical locations of q0 and q1
            p0 = np.where(self.layout == q0)[0][0]
            p1 = np.where(self.layout == q1)[0][0]
            
            # Are they adjacent on the path graph?
            if abs(p0 - p1) == 1:
                self.gate_index += 1
                executed_count += 1
            else:
                break # Need a SWAP
        return executed_count

    def step(self, action):
        self.current_step += 1
        reward = 0
        
        # Action 0 -> Swap 0 & 1
        # Action 1 -> Swap 1 & 2
        # Action 2 -> Swap 2 & 3
        # Action 3 -> Swap 3 & 4
        p_a = action
        p_b = action + 1
        
        # Apply the physical SWAP
        self.layout[p_a], self.layout[p_b] = self.layout[p_b], self.layout[p_a]
        
        # Penalize for taking a SWAP
        reward -= 1.0

        # Try to execute gates now that we swapped
        executed = self._auto_execute()
        
        # Massive reward for executing a gate (encourages finding the shortest path of SWAPs)
        reward += executed * 10.0

        # Check if done
        terminated = bool(self.gate_index >= len(self.gates))
        truncated = bool(self.current_step >= self.max_steps)

        # If we hit the step limit and didn't finish, heavy penalty
        if truncated and not terminated:
            reward -= 50.0

        return self._get_obs(), reward, terminated, truncated, {}
