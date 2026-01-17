import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
train_df = pd.read_csv("data/raw/train.csv")
features_df = pd.read_csv("data/raw/features.csv")

# Basic inspection
print(train_df.head())
print(train_df.info())
print(train_df.describe())
print(train_df.isnull().sum())

# Convert Date column to datetime
train_df["Date"] = pd.to_datetime(train_df["Date"])

print(train_df.dtypes)

# Feature engineering from Date
train_df["Year"] = train_df["Date"].dt.year
train_df["Month"] = train_df["Date"].dt.month
train_df["Week"] = train_df["Date"].dt.isocalendar().week

print(train_df[["Date", "Year", "Month", "Week"]].head())

# Aggregate weekly sales over time
sales_over_time = train_df.groupby("Date")["Weekly_Sales"].sum()

plt.figure(figsize=(12,5))
plt.plot(sales_over_time)
plt.title("Total Weekly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.tight_layout()
plt.show()

holiday_sales = train_df.groupby("IsHoliday")["Weekly_Sales"].mean()
print(holiday_sales)
