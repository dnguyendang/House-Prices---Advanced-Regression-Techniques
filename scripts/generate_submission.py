import pandas as pd
import joblib

# Load test data
test_df = pd.read_csv("../data/test.csv")

# Load trained model
model = joblib.load("../results/best_model.pkl")

# Predict house prices
predictions = model.predict(test_df)

# Save submission file
submission = pd.DataFrame({"Id": test_df["Id"], "SalePrice": predictions})
submission.to_csv("../results/submission.csv", index=False)

print("Submission file generated: results/submission.csv")