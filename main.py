from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- DATA LOADING ---
car_evaluation = fetch_ucirepo(id=19)
X = car_evaluation.data.features
y = car_evaluation.data.targets['class'] 

# --- ALGORITHM 1: STRICT BOOLEAN EXPERT SYSTEM ---
def boolean_expert_system(row):
    safety = row['safety']
    persons = row['persons']
    buying = row['buying']
    maint = row['maint']
    
    if safety == 'low': return 'unacc'
    elif persons == '2': return 'unacc'
    elif safety == 'high' and buying == 'low' and maint == 'low': return 'vgood'
    elif safety in ['high', 'med'] and buying in ['low', 'med'] and maint in ['low', 'med']: return 'good'
    else: return 'acc'

predictions_alg1 = X.apply(boolean_expert_system, axis=1)

# --- ALGORITHM 2: WEIGHTED SCORING RULE-BASED SYSTEM ---
def weighted_rule_system(row):
    score = 0
    if row['safety'] == 'high': score += 4
    elif row['safety'] == 'med': score += 2
    elif row['safety'] == 'low': score -= 10 
        
    if row['persons'] == 'more': score += 2
    elif row['persons'] == '4': score += 1
    elif row['persons'] == '2': score -= 10 
        
    if row['buying'] in ['low', 'med']: score += 1
    if row['maint'] in ['low', 'med']: score += 1

    if score < 0: return 'unacc'
    elif score >= 0 and score < 5: return 'acc'
    elif score >= 5 and score <= 6: return 'good'
    else: return 'vgood'

predictions_alg2 = X.apply(weighted_rule_system, axis=1)

# --- RESULTS, METRICS & IMAGES ---
classes = ['unacc', 'acc', 'good', 'vgood']

print("=== ALGORITHM 1: BOOLEAN EXPERT SYSTEM ===")
print(f"Accuracy: {accuracy_score(y, predictions_alg1) * 100:.2f}%\n")
print(classification_report(y, predictions_alg1, zero_division=0))

# Generate and save Alg 1 Confusion Matrix
cm1 = confusion_matrix(y, predictions_alg1, labels=classes)
disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=classes)
disp1.plot(cmap=plt.cm.Blues)
plt.title("Algorithm 1: Boolean Expert System")
plt.savefig("alg1_matrix.png", bbox_inches='tight', dpi=300)
plt.close()

print("=== ALGORITHM 2: WEIGHTED SCORING SYSTEM ===")
print(f"Accuracy: {accuracy_score(y, predictions_alg2) * 100:.2f}%\n")
print(classification_report(y, predictions_alg2, zero_division=0))

# Generate and save Alg 2 Confusion Matrix
cm2 = confusion_matrix(y, predictions_alg2, labels=classes)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=classes)
disp2.plot(cmap=plt.cm.Oranges)
plt.title("Algorithm 2: Weighted Scoring System")
plt.savefig("alg2_matrix.png", bbox_inches='tight', dpi=300)
plt.close()

print("Images 'alg1_matrix.png' and 'alg2_matrix.png' have been saved to your folder!")