from ucimlrepo import fetch_ucirepo
import pandas as pd

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
    
    # Rule Set 1: Safety Weights (Safety is paramount)
    if row['safety'] == 'high': score += 4
    elif row['safety'] == 'med': score += 2
    elif row['safety'] == 'low': score -= 10 # Critical penalty
        
    # Rule Set 2: Capacity Weights
    if row['persons'] == 'more': score += 2
    elif row['persons'] == '4': score += 1
    elif row['persons'] == '2': score -= 10 # Critical penalty
        
    # Rule Set 3: Economic Weights (Lower costs add points)
    if row['buying'] in ['low', 'med']: score += 1
    if row['maint'] in ['low', 'med']: score += 1

    # Inference Engine: Score Thresholds
    if score < 0:
        return 'unacc'
    elif score >= 0 and score < 5:
        return 'acc'
    elif score >= 5 and score <= 6:
        return 'good'
    else:
        return 'vgood'

predictions_alg2 = X.apply(weighted_rule_system, axis=1)

# --- RESULTS & METRICS ---
def calculate_accuracy(predictions, actual):
    correct = (predictions == actual).sum()
    return (correct / len(actual)) * 100

print("==================================================")
print("ALGORITHM 1: BOOLEAN EXPERT SYSTEM")
print(f"Accuracy: {calculate_accuracy(predictions_alg1, y):.2f}%")
print("==================================================")
print("ALGORITHM 2: WEIGHTED SCORING SYSTEM")
print(f"Accuracy: {calculate_accuracy(predictions_alg2, y):.2f}%")
print("==================================================")