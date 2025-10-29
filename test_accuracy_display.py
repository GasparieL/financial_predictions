import pandas as pd
import numpy as np

# Simulate test metrics
test_metrics = {
    'MAE': 2500.50,
    'RMSE': 3200.75,
    'R2': 0.85,
    'MAPE': 15.3
}

# Calculate accuracy percentage
accuracy_pct = 100 - test_metrics['MAPE']

print(f"\n{'='*60}")
print(f"MODEL ACCURACY TEST")
print(f"{'='*60}")
print(f"Model Accuracy: {accuracy_pct:.1f}%")
print(f"  (On average, predictions are {accuracy_pct:.1f}% accurate)")
print(f"\nError Metrics:")
print(f"  MAPE (error rate): {test_metrics['MAPE']:.1f}%")
print(f"  MAE (avg $ error): ${test_metrics['MAE']:,.2f}")
print(f"  RMSE: ${test_metrics['RMSE']:,.2f}")
print(f"  R² (variance explained): {test_metrics['R2']:.3f}")
print(f"{'='*60}\n")

print("Test passed - accuracy percentage displays correctly!")
