"""
OPTIONAL ML upgrade path.

The rule-based system in flood_risk.py works fine on its own and is
what I'd recommend demoing first. This script shows how you'd
upgrade to an ML model IF you get access to real historical flood
event records (did village X flood on date Y, given the conditions
that day).

Since we don't have real historical flood records, this script
generates synthetic "did it flood" labels using a noisy version of
the same rule-based logic — purely to demonstrate the ML pipeline
(train/test split, model, evaluation). Swap generate_synthetic_labels()
for real historical data before using this for anything but a demo.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from flood_risk import compute_risk


def generate_synthetic_training_data(n_samples=2000, seed=0):
    """
    Creates synthetic (features, flooded_or_not) pairs for demo purposes.
    Replace this with real historical event data for an actual deployment.
    """
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "slope_deg": rng.uniform(2, 45, n_samples),
        "recent_rainfall_3day_mm": rng.uniform(0, 150, n_samples),
        "distance_to_stream_m": rng.uniform(50, 3000, n_samples),
    })
    current_rainfall = pd.Series(rng.uniform(0, 80, n_samples))

    scored = compute_risk(df, current_rainfall)

    # Simulate real-world noise: high risk score usually (not always) floods
    flood_prob = (scored["risk_score"] / 100).clip(0, 1)
    scored["flooded"] = (rng.uniform(0, 1, n_samples) < flood_prob).astype(int)

    return scored


def main():
    data = generate_synthetic_training_data()

    features = [
        "slope_deg",
        "recent_rainfall_3day_mm",
        "distance_to_stream_m",
        "current_rainfall_mm_per_hr",
    ]
    X = data[features]
    y = data["flooded"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Model evaluation on held-out synthetic test set:")
    print(classification_report(y_test, preds, target_names=["No Flood", "Flood"]))

    print("\nFeature importances:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {feat}: {imp:.3f}")

    joblib.dump(model, "flood_model.joblib")
    print("\nSaved trained model to flood_model.joblib")


if __name__ == "__main__":
    main()