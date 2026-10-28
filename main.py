from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

# --- DATA LOADING ---
car_evaluation = fetch_ucirepo(id=19)
X = car_evaluation.data.features
y = car_evaluation.data.targets['class'] 

# --- ALGORITHM 1: MANUAL EXPERT SYSTEM ---
def manual_expert_system(row):
    safety = row['safety']
    persons = row['persons']
    buying = row['buying']
    maint = row['maint']
    
    if safety == 'low': return 'unacc'
    elif persons == '2': return 'unacc'
    elif safety == 'high' and buying == 'low' and maint == 'low': return 'vgood'
    elif safety in ['high', 'med'] and buying in ['low', 'med'] and maint in ['low', 'med']: return 'good'
    else: return 'acc'

predictions_alg1 = X.apply(manual_expert_system, axis=1)

# --- ALGORITHM 2: INDUCTIVE RULE-BASED SYSTEM (DECISION TREE) ---
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

inductive_tree = DecisionTreeClassifier(max_depth=4, random_state=42)
inductive_tree.fit(X_encoded, y)
predictions_alg2 = inductive_tree.predict(X_encoded)

# --- RESULTS & METRICS ---
print("==================================================")
print("ALGORITHM 1: MANUAL EXPERT SYSTEM RESULTS")
print("==================================================")
print(f"Accuracy: {accuracy_score(y, predictions_alg1) * 100:.2f}%\n")
print(classification_report(y, predictions_alg1, zero_division=0))

print("\n==================================================")
print("ALGORITHM 2: INDUCTIVE RULE-BASED SYSTEM RESULTS")
print("==================================================")
print(f"Accuracy: {accuracy_score(y, predictions_alg2) * 100:.2f}%\n")
print(classification_report(y, predictions_alg2, zero_division=0))