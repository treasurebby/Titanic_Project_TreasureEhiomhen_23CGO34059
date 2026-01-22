# model_building.py
# Titanic Survival Prediction - Model Development

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

print("=" * 60)
print("TITANIC SURVIVAL PREDICTION - MODEL DEVELOPMENT")
print("=" * 60)

# Step 1: Load the Titanic dataset
print("\n[1] Loading Titanic Dataset...")
try:
    # Try to load from local file first
    df = pd.read_csv('titanic.csv')
except FileNotFoundError:
    # If not found, load from seaborn
    import seaborn as sns
    df = sns.load_dataset('titanic')
    df.to_csv('titanic.csv', index=False)

print(f"Dataset loaded successfully! Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

# Step 2: Data Preprocessing
print("\n" + "=" * 60)
print("[2] DATA PREPROCESSING")
print("=" * 60)

# Check for missing values
print("\nMissing values before preprocessing:")
print(df.isnull().sum())

# Select the 5 required features + target variable
# Selected features: Pclass, Sex, Age, Fare, Embarked
selected_features = ['pclass', 'sex', 'age', 'fare', 'embarked', 'survived']

# Handle column name variations
df.columns = df.columns.str.lower()
if 'pclass' not in df.columns and 'class' in df.columns:
    df['pclass'] = df['class']

# Filter only selected columns
df = df[selected_features].copy()

print(f"\nSelected Features: {selected_features[:-1]}")
print(f"Target Variable: survived")

# Step 2a: Handling Missing Values
print("\n[2a] Handling Missing Values...")

# Fill missing Age values with median
df['age'].fillna(df['age'].median(), inplace=True)

# Fill missing Fare values with median
df['fare'].fillna(df['fare'].median(), inplace=True)

# Fill missing Embarked values with mode
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# Drop any remaining rows with missing values
df.dropna(inplace=True)

print("Missing values after preprocessing:")
print(df.isnull().sum())

# Step 2b & 2c: Feature Selection and Encoding Categorical Variables
print("\n[2b & 2c] Encoding Categorical Variables...")

# Create encoders
le_sex = LabelEncoder()
le_embarked = LabelEncoder()

# Encode Sex (male=1, female=0)
df['sex_encoded'] = le_sex.fit_transform(df['sex'])

# Encode Embarked
df['embarked_encoded'] = le_embarked.fit_transform(df['embarked'])

print(f"Sex encoding: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
print(f"Embarked encoding: {dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))}")

# Prepare features and target
X = df[['pclass', 'sex_encoded', 'age', 'fare', 'embarked_encoded']]
y = df['survived']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Step 2d: Feature Scaling
print("\n[2d] Feature Scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("Features scaled successfully!")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Step 3: Implement Machine Learning Algorithm
print("\n" + "=" * 60)
print("[3] TRAINING RANDOM FOREST CLASSIFIER")
print("=" * 60)

# Initialize and train Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
rf_model.fit(X_train, y_train)
print("Model trained successfully!")

# Step 4: Evaluate the Model
print("\n" + "=" * 60)
print("[4] MODEL EVALUATION")
print("=" * 60)

# Make predictions
y_pred = rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Did Not Survive', 'Survived']))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\n[[TN  FP]")
print(" [FN  TP]]")

# Feature Importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance)

# Step 5: Save the Model
print("\n" + "=" * 60)
print("[5] SAVING MODEL AND PREPROCESSING OBJECTS")
print("=" * 60)

# Save all necessary objects
joblib.dump(rf_model, 'model/titanic_survival_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le_sex, 'model/le_sex.pkl')
joblib.dump(le_embarked, 'model/le_embarked.pkl')

print("\nSaved objects:")
print("✓ model/titanic_survival_model.pkl")
print("✓ model/scaler.pkl")
print("✓ model/le_sex.pkl")
print("✓ model/le_embarked.pkl")

# Step 6: Demonstrate Model Loading and Prediction
print("\n" + "=" * 60)
print("[6] TESTING SAVED MODEL - RELOAD AND PREDICT")
print("=" * 60)

# Load the saved model
loaded_model = joblib.load('model/titanic_survival_model.pkl')
loaded_scaler = joblib.load('model/scaler.pkl')
loaded_le_sex = joblib.load('model/le_sex.pkl')
loaded_le_embarked = joblib.load('model/le_embarked.pkl')

print("\nModel and preprocessors loaded successfully!")

# Test prediction with sample data
sample_data = {
    'Pclass': 3,
    'Sex': 'male',
    'Age': 22,
    'Fare': 7.25,
    'Embarked': 'S'
}

print(f"\nSample Passenger Data:")
for key, value in sample_data.items():
    print(f"  {key}: {value}")

# Prepare sample for prediction
sex_encoded = loaded_le_sex.transform([sample_data['Sex']])[0]
embarked_encoded = loaded_le_embarked.transform([sample_data['Embarked']])[0]

sample_features = np.array([[
    sample_data['Pclass'],
    sex_encoded,
    sample_data['Age'],
    sample_data['Fare'],
    embarked_encoded
]])

# Scale features
sample_scaled = loaded_scaler.transform(sample_features)

# Make prediction
prediction = loaded_model.predict(sample_scaled)[0]
probability = loaded_model.predict_proba(sample_scaled)[0]

print(f"\nPrediction Result:")
print(f"  Survived: {'Yes' if prediction == 1 else 'No'}")
print(f"  Probability [Not Survive, Survive]: [{probability[0]:.4f}, {probability[1]:.4f}]")
print(f"  Confidence: {max(probability)*100:.2f}%")

print("\n" + "=" * 60)
print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\nNext Steps:")
print("1. Run app.py to start the Flask web application")
print("2. Open browser and navigate to http://localhost:5000")
print("3. Test the prediction system with different passenger data")
print("=" * 60)