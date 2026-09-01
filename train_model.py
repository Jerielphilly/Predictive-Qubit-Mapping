import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


print("Training regression model...")


df = pd.read_csv(
    "dataset.csv"
)


FEATURES = [
    "distance",
    "upcoming_gates",
    "remaining_gates",
    "physical_q0",
    "physical_q1",
    "current_swap_count",
    "candidate_swap_a",
    "candidate_swap_b"
]


X = df[FEATURES]

y = df[
    "future_swap_cost"
]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=1
)


model.fit(
    X_train,
    y_train
)


predictions = model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    predictions
)


mse = mean_squared_error(
    y_test,
    predictions
)


r2 = r2_score(
    y_test,
    predictions
)


print()
print("Model training complete!")
print()

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)

print(
    "MAE:",
    round(mae, 4)
)

print(
    "MSE:",
    round(mse, 4)
)

print(
    "R2 Score:",
    round(r2, 4)
)


joblib.dump(
    model,
    "routing_model.pkl"
)


print()
print("Model saved as routing_model.pkl")