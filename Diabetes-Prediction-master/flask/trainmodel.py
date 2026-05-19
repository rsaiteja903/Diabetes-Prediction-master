import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# 1. Load the diabetes dataset from the original GitHub source
DATA_URL = "https://raw.githubusercontent.com/Abhayparashar31/Diabetes-prediction/master/diabetes.csv"
print("Loading dataset...")
data = pd.read_csv(DATA_URL)

# 2. Split into features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# 3. Build a simple pipeline: StandardScaler + SVC
model = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("svc", SVC(probability=True, random_state=42)),
    ]
)

print("Training model...")
model.fit(X, y)

# 4. Save the trained model as model.pkl (overwrite old one)
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ New model.pkl saved successfully!")