import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load data
df = pd.read_csv("data/raw/train.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"])

# Feature engineering (same logic as EDA)
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# Select features and target
features = ["Store", "Dept", "Year", "Month", "Week", "IsHoliday"]
X = df[features]
y = df["Weekly_Sales"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Baseline Linear Regression Results")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")