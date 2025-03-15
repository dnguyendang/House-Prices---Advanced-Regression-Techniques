import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Ensure dataset exists
train_path = "../data/train.csv"
if not os.path.exists(train_path):
    raise FileNotFoundError(f"Dataset not found: {train_path}")

# Load dataset
train_df = pd.read_csv(train_path)

# Ensure 'SalePrice' exists
if "SalePrice" not in train_df.columns:
    raise ValueError("Column 'SalePrice' not found in dataset")

# Identify numerical and categorical features
num_features = train_df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = train_df.select_dtypes(include=["object"]).columns.tolist()

# Remove 'SalePrice' from feature lists
num_features.remove("SalePrice") 

# Log-transform SalePrice to stabilize variance
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

# Preprocessing Pipelines
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())  # Normalize numerical data
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine pipelines
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# Prepare features and target
X = train_df.drop(columns=["SalePrice"])
y = train_df["SalePrice"]

# Apply preprocessing
X_transformed = preprocessor.fit_transform(X)

# Split dataset
X_train, X_valid, y_train, y_valid = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Print target variable stats
print(f"Min SalePrice: {y_train.min():.2f}, Max: {y_train.max():.2f}, Mean: {y_train.mean():.2f}")

# Train models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_rmse = float("inf")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    
    # Convert RMSE back to original scale
    rmse = np.sqrt(mean_squared_error(np.expm1(y_valid), np.expm1(preds)))
    print(f"{name} - Log Scale RMSE: {np.sqrt(mean_squared_error(y_valid, preds)):.4f}")
    print(f"{name} - Real Scale RMSE: {np.sqrt(mean_squared_error(np.expm1(y_valid), np.expm1(preds))):.2f}")
    # Select the best model
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_model_name = name

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Save best model
if best_model:
    model_path = "/results/best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"Best model '{best_model_name}' saved to {model_path} with RMSE: {best_rmse:.4f}")
else:
    print("No valid model found.")
