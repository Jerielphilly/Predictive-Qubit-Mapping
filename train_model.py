import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

FEATURES = [
    "distance", "upcoming_gates", "remaining_gates", 
    "physical_q0", "physical_q1", 
    "current_swap_count", "candidate_swap_a", "candidate_swap_b"
]

print("Loading dataset...")
data_path = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
df = pd.read_csv(data_path)

X = df[FEATURES]
y = df["target_score"]

print(f"Dataset size: {len(df)} samples")
print("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)

model.fit(X, y)

print("Training complete! Saving model...")
model_path = os.path.join(os.path.dirname(__file__), "models", "routing_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved as '{model_path}'")