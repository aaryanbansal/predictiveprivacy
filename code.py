import numpy as np
import pandas as pd

# Load a small portion of the UCI Heart Disease dataset with only cholesterol (chol) and age for simplicity
data = {'age': [63, 67, 67, 37, 41],
        'chol': [233, 286, 229, 250, 204]}  # Cholesterol levels (example values)
df = pd.DataFrame(data)

print("Original Data:")
print(df)

# Differential Privacy: Laplace Mechanism function
def apply_laplace_mechanism(value, sensitivity, epsilon):
    """Applies the Laplace mechanism to the given value."""
    noise = np.random.laplace(0, sensitivity / epsilon)
    return value + noise

# Parameters
epsilon = 0.5  # Privacy budget; adjust for stronger or weaker privacy
sensitivity = 1  # Assume sensitivity = 1 for this example

# Apply differential privacy (Laplace mechanism) to the cholesterol column
df['chol_noisy'] = df['chol'].apply(lambda x: apply_laplace_mechanism(x, sensitivity, epsilon))

print("\nData After Applying Differential Privacy (Noisy Cholesterol Levels):")
print(df)
