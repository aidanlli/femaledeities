import pandas as pd
from fuzzywuzzy import process, fuzz
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# File paths
truth_path = r"C:\Users\aidan\Downloads\filtered_250_sampled_rows.csv"
final_path = r"C:\Users\aidan\Downloads\4.final.csv"

# Load datasets
truth_df = pd.read_csv(truth_path, encoding="utf-8-sig")
final_df = pd.read_csv(final_path, encoding="utf-8-sig")

# Function to normalize deity names
def normalize_name(name):
    if pd.isna(name) or name.lower() == "missing":
        return None
    return name.lower().strip()

# Function to match deities using fuzzy matching
def match_deities(truth_deities, final_deities):
    final_deities_set = set(final_deities) if final_deities else set()
    matches = {}

    for deity in truth_deities:
        if deity and isinstance(deity, str):  # Ensure deity is a string
            match, score = process.extractOne(deity, final_deities_set, scorer=fuzz.token_sort_ratio) or ("missing", 0)
        else:
            match, score = "missing", 0
        matches[deity] = (match, score)
    
    return matches

# Processing each row
results = []
for _, truth_row in truth_df.iterrows():
    truth_deities = [normalize_name(d) for d in str(truth_row.get('Deities', 'missing')).split(', ')]

    uuid = truth_row['uuid']
    truth_deities = [normalize_name(d) for d in str(truth_row['Deities']).split(', ')]
    final_row = final_df[final_df['uuid'] == uuid]
    
    if final_row.empty:
        continue
    
    final_deities = [normalize_name(d) for d in str(final_row['deities'].values[0]).split(', ')]
    matched_deities = match_deities(truth_deities, final_deities)
    
    # Compute score
    score = [1 if match and match[1] else 0 for match in matched_deities]
    
    results.append({"uuid": uuid, "score": score})

# Convert results to dataframe
results_df = pd.DataFrame(results)

# Compute overall metrics
y_true = [1 for row in results for _ in row["score"]]
y_pred = [val for row in results for val in row["score"]]
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
accuracy = accuracy_score(y_true, y_pred)

# Print results
print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, Accuracy: {accuracy:.2f}")

# Save results
results_df.to_csv(r"C:\Users\aidan\Downloads\deity_matching_results.csv", index=False, encoding="utf-8-sig")
