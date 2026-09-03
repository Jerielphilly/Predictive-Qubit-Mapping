import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from rl_env import QuantumRoutingEnv

print("Initializing RL Environment...")
env = QuantumRoutingEnv()

# Validate the custom environment follows gymnasium standards
check_env(env, warn=True)

print("Environment validated successfully!")
print("Starting Proximal Policy Optimization (PPO) Training...")

# Set up an evaluation callback to save the best model during training
eval_env = QuantumRoutingEnv()
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path='./models/',
    log_path='./logs/', 
    eval_freq=10000,
    deterministic=True, 
    render=False
)

# Initialize PPO Agent
# MlpPolicy is a standard dense neural network (Multi-Layer Perceptron)
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, n_steps=2048)

# Train the agent (1,000,000 timesteps is a good start for a complex RL task)
# For a quick test, you can reduce this to 50,000
TOTAL_TIMESTEPS = 100_000 

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

# Save the final model
model.save("ppo_quantum_router")
print(f"Training complete! Model saved as 'ppo_quantum_router.zip'")
