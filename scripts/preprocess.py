import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Load dataset
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

# Define numerical and categorical columns
num_features = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = train_df.select_dtypes(include=['object']).columns.tolist()

# Ensure 'SalePrice' is not in feature lists (only applies to train_df)
if 'SalePrice' in num_features:
    num_features.remove('SalePrice')

# Pipelines for preprocessing
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Full transformation pipeline
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', cat_pipeline, cat_features)
])

# Prepare training data
X_train = train_df.drop(columns=['SalePrice'])
y_train = train_df['SalePrice']
X_train_transformed = preprocessor.fit_transform(X_train)

# Prepare test data (no 'SalePrice' column)
X_test_transformed = preprocessor.transform(test_df)

print("Data Preprocessing Completed.")