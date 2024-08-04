import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Example data for scaler generation
data = {
    'feature1': [1, 2, 3, 4, 5],
    'feature2': [10, 20, 30, 40, 50],
    # Add more features if needed
}
df = pd.DataFrame(data)

# Fit the scaler on the data
scaler = StandardScaler()
scaler.fit(df)

# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')
