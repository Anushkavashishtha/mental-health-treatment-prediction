# train.py
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and clean data
df = pd.read_csv("survey.csv")
df = df.dropna(subset=["treatment","work_interfere","benefits","leave","self_employed","family_history"])
df = df[df["Age"].between(18,70)]

# Encode features
cols = ["Gender","family_history","work_interfere","self_employed","benefits","leave","phys_health_consequence"]
le = LabelEncoder()
for c in cols:
    df[c] = le.fit_transform(df[c].astype(str))

X = df[cols + ["Age"]]
y = le.fit_transform(df["treatment"].astype(str))  # yes=1, no=0

# Split, train, evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/mh_model.pkl")
