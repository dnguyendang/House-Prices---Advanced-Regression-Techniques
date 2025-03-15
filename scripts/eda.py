import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Load dataset
train_df = pd.read_csv("../data/train.csv")

# Display dataset info
print(train_df.info())

# Missing values visualization
plt.figure(figsize=(12, 6))
msno.matrix(train_df)
plt.title("Missing Data Heatmap")
plt.show()

# SalePrice distribution
plt.figure(figsize=(8, 5))
sns.histplot(train_df["SalePrice"], kde=True, bins=30)
plt.title("SalePrice Distribution")
plt.show()

# Take only numerical columns
numeric_features = train_df.select_dtypes(include=[np.number])
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_features.corr(), cmap="coolwarm", annot=False, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()