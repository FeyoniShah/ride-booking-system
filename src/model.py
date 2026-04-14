import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# -----------------------------
# LOAD DATA (ROBUST PATH)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, 'cleaned_data.csv')

df = pd.read_csv(data_path)

print("Data loaded successfully:", df.shape)


# -----------------------------
# TARGET VARIABLE
# -----------------------------
# 1 = Cancelled, 0 = Not Cancelled
df['Cancelled'] = df['Booking Status'].apply(
    lambda x: 1 if 'Cancelled' in str(x) else 0
)

print("\nTarget distribution:")
print(df['Cancelled'].value_counts())


# -----------------------------
# SELECT FEATURES
# -----------------------------
# features = [
#     'Avg VTAT',
#     'Avg CTAT',
#     'Ride Distance',
#     'Hour',
#     'Vehicle Type'
# ]



# features = [
#     'Hour',
#     'Vehicle Type',
#     'Pickup Location',
#     'Drop Location'
# ]

features = [
    'Hour',
    'Vehicle Type',
    'Pickup Location',
    'Drop Location',
    'Ride Distance',
    'Avg VTAT',
    'Avg CTAT'
]

df_model = df[features + ['Cancelled']].copy()


# -----------------------------
# HANDLE MISSING VALUES (IMPORTANT)
# -----------------------------
# Do NOT drop rows (prevents losing cancelled rides)

df_model['Ride Distance'] = df_model['Ride Distance'].fillna(0)

df_model['Avg VTAT'] = df_model['Avg VTAT'].fillna(
    df_model['Avg VTAT'].median()
)

df_model['Avg CTAT'] = df_model['Avg CTAT'].fillna(
    df_model['Avg CTAT'].median()
)


# -----------------------------
# ENCODE CATEGORICAL
# -----------------------------
le_vehicle = LabelEncoder()
le_pickup = LabelEncoder()
le_drop = LabelEncoder()

df_model['Vehicle Type'] = le_vehicle.fit_transform(df_model['Vehicle Type'])
df_model['Pickup Location'] = le_pickup.fit_transform(df_model['Pickup Location'])
df_model['Drop Location'] = le_drop.fit_transform(df_model['Drop Location'])

# -----------------------------
# SPLIT DATA (STRATIFIED)
# -----------------------------
X = df_model[features]
y = df_model['Cancelled']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y   # IMPORTANT: keeps class balance
)


# -----------------------------
# MODEL TRAINING
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------------
# EVALUATION
# -----------------------------
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))


from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

# -----------------------------
# SAVE MODEL
# -----------------------------
model_dir = os.path.join(BASE_DIR, 'model')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
joblib.dump({
    'vehicle': le_vehicle,
    'pickup': le_pickup,
    'drop': le_drop
}, os.path.join(model_dir, 'encoders.pkl'))

print("\nModel and encoder saved successfully!")


