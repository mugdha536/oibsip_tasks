import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
file_path = "D:\\My Data\\Mugdha jadhav\\Desktop\\oasis internship\\fraud detection (credit card) task 3\\creditcard.csv"
df = pd.read_csv(file_path)

# Reduce dataset size before SMOTE
df_majority = df[df["Class"] == 0].sample(n=50000, random_state=42)  # Sample 50,000 non-fraud cases
df_minority = df[df["Class"] == 1]  # Keep all fraud cases

df_balanced = pd.concat([df_majority, df_minority])

# Split features and target
X = df_balanced.drop(columns=["Class"])
y = df_balanced["Class"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Apply limited SMOTE
smote = SMOTE(sampling_strategy=0.1, random_state=42)  # Create 10% synthetic fraud cases
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Train Random Forest model with optimized parameters
rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_smote, y_train_smote)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate model
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Random Forest")
plt.show()
