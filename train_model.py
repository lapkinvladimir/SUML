from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split data into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print accuracy
print(f"Model accuracy: {accuracy:.2f}")

# Ensure 'app' directory exists
os.makedirs("app", exist_ok=True)

# Save the model
joblib.dump(model, "app/model.joblib")

print("Model saved to app/model.joblib")
