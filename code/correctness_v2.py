import pandas as pd
from fuzzywuzzy import process, fuzz
from sklearn.metrics import accuracy_score
import ast
import re
# File paths
truth_path = r"C:\Users\aidan\Downloads\filtered_250_sampled_rows.csv"
final_path = r"C:\Users\aidan\Downloads\3.final.csv"

# Load datasets
truth_df = pd.read_csv(truth_path, encoding="utf-8-sig")
final_df = pd.read_csv(final_path, encoding="utf-8-sig")

# Normalize function
def normalize_name(name):
    if pd.isna(name) or str(name).strip().lower() in {"", "missing"}:
        return None
    return str(name).strip().lower()

# Normalize gender column in final_df
final_df['gender'] = final_df['gender'].str.strip().str.lower().replace({'male & female': 'general'})

# Parse helper functions
def parse_comma_column(column):
    if isinstance(column, str):
        # Heuristic: looks like a list if it starts with [ and ends with ]
        if column.strip().startswith("[") and column.strip().endswith("]"):
            try:
                evaluated = ast.literal_eval(column)
                if isinstance(evaluated, list):
                    return [normalize_name(item) for item in evaluated]
            except (ValueError, SyntaxError):
                pass  # Fall through to comma split

        # Fallback: treat as normal text, split by commas
        return [normalize_name(part) for part in column.split(',')]
    return []

def parse_linked_column(column):
    return [normalize_name(part) for part in column.split(';')] if isinstance(column, str) else []

# Match deities with aliases
def match_deities_with_aliases(deity_aliases, final_deities):
    candidate_matches = []

    # Step 1: Collect all possible matches with scores
    for base_deity, aliases in deity_aliases.items():
        for alias in aliases:
            match, score = process.extractOne(alias, final_deities, scorer=fuzz.token_sort_ratio) or (None, 0)
            if match:
                candidate_matches.append({
                    "base_deity": base_deity,
                    "alias": alias,
                    "match": match,
                    "score": score
                })

    # Step 2: Sort matches by score descending
    candidate_matches.sort(key=lambda x: x["score"], reverse=True)

    # Step 3: Assign matches greedily ensuring one-to-one matching
    used_final_deities = set()
    matched_final_set = set()
    matches = {}

    for candidate in candidate_matches:
        base = candidate["base_deity"]
        match = candidate["match"]
        if base in matches:
            continue  # already matched
        if match in used_final_deities:
            continue  # already used by another base_deity
        matches[base] = {
            "match": match,
            "score": candidate["score"],
            "matched_via": candidate["alias"]
        }
        used_final_deities.add(match)
        matched_final_set.add(match)

    # Step 4: Fill in unmatched deities as "missing"
    for base in deity_aliases:
        if base not in matches:
            matches[base] = {
                "match": "missing",
                "score": 0,
                "matched_via": None
            }

    return matches


# Evaluation function
def evaluate_row(uuid, truth_row, final_row):
    truth_deities = parse_comma_column(truth_row.get('Deities', ''))
    truth_genders = parse_comma_column(truth_row.get('Gender', ''))
    other_names = parse_linked_column(truth_row.get('Other_names', ''))

    final_deities = parse_comma_column(final_row.get('deities', ''))
    final_genders = parse_comma_column(final_row.get('gender', ''))

    deity_aliases = {}
    for i, deity in enumerate(truth_deities):
        if deity is None:
            continue
        alias = other_names[i] if i < len(other_names) else None
        deity_aliases[deity] = [deity] + ([alias] if alias else [])

    if all(d is None for d in truth_deities) and all(d is None for d in final_deities):
        return {
            "uuid": uuid,
            "truth_deities": truth_row.get('Deities', ''),
            "final_deities": final_row.get('deities', ''),
            "matched_deities": str({"missing": ("missing", 100)}),
            "score": [1],
            "gender_score": [1]
        }

    elif all(d is None for d in truth_deities):
        unmatched = [d for d in final_deities if d]
        score = [0 for _ in unmatched]
        return {
            "uuid": uuid,
            "truth_deities": truth_row.get('Deities', ''),
            "final_deities": final_row.get('deities', ''),
            "matched_deities": str({"missing": (", ".join(unmatched), 0)}),
            "score": score,
            "gender_score": [0] * len(score)
        }

    elif any(d is not None for d in truth_deities) and all(d is None for d in final_deities):
        score = [0 for d in truth_deities if d is not None]
        return {
            "uuid": uuid,
            "truth_deities": truth_row.get('Deities', ''),
            "final_deities": final_row.get('deities', ''),
            "matched_deities": str({d: ("missing", 0) for d in truth_deities if d}),
            "score": score,
            "gender_score": [0] * len(score)
        }

    matched_deities = match_deities_with_aliases(deity_aliases, final_deities)
    score = []
    gender_score = []
    used_final_deities = set()

    for i, deity in enumerate(truth_deities):
        if deity is None:
            continue
        match_info = matched_deities.get(deity, {"match": "missing", "score": 0})
        matched_deity = match_info["match"]
        match_score = match_info["score"]

        found = match_score >= 50
        score.append(1 if found else 0)

        if found:
            used_final_deities.add(matched_deity)

        if found and i < len(truth_genders):
            truth_gender = truth_genders[i]
            try:
                final_idx = final_deities.index(matched_deity)
                final_gender = final_genders[final_idx] if final_idx < len(final_genders) else None
                gender_score.append(1 if truth_gender == final_gender else 0)
            except ValueError:
                gender_score.append(0)
        else:
            gender_score.append(0)

    # Identify false positives — unmatched predicted deities
    unmatched_predictions = [d for d in final_deities if d not in used_final_deities]
    score += [0 for _ in unmatched_predictions]  # Truth = 0, Predicted = 1
    gender_score += [0 for _ in unmatched_predictions]  # No gender match

    return {
        "uuid": uuid,
        "truth_deities": truth_row.get('Deities', ''),
        "truth_genders": truth_row.get('Gender', ''),
        "other_names": truth_row.get('Other_names', ''),
        "final_deities": final_row.get('deities', ''),
        "final_genders": final_row.get('gender', ''),
        "matched_deities": str(matched_deities),
        "score": score,
        "gender_score": gender_score
    }


# Run evaluation
results = []
for _, truth_row in truth_df.iterrows():
    uuid = truth_row['uuid']
    final_row = final_df[final_df['uuid'] == uuid]
    if final_row.empty:
        continue
    result = evaluate_row(uuid, truth_row, final_row.iloc[0])
    results.append(result)

# Convert results
results_df = pd.DataFrame(results)

# Metrics
y_true = [1 for row in results for _ in row["score"]]
y_pred = [val for row in results for val in row["score"]]

gender_true = [1 for row in results for _ in row["gender_score"]]
gender_pred = [val for row in results for val in row["gender_score"]]

# Deity metrics
TP = sum(p == 1 and t == 1 for p, t in zip(y_pred, y_true))
FP = sum(p == 1 and t == 0 for p, t in zip(y_pred, y_true))
FN = sum(p == 0 and t == 1 for p, t in zip(y_pred, y_true))
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
accuracy = accuracy_score(y_true, y_pred)

# Gender metrics
gTP = sum(p == 1 and t == 1 for p, t in zip(gender_pred, gender_true))
gFP = sum(p == 1 and t == 0 for p, t in zip(gender_pred, gender_true))
gFN = sum(p == 0 and t == 1 for p, t in zip(gender_pred, gender_true))
g_precision = gTP / (gTP + gFP) if (gTP + gFP) else 0
g_recall = gTP / (gTP + gFN) if (gTP + gFN) else 0
g_f1 = 2 * g_precision * g_recall / (g_precision + g_recall) if (g_precision + g_recall) else 0
g_accuracy = accuracy_score(gender_true, gender_pred)

# Print results
print("Deity Matching:")
print(f"  TP: {TP}, FP: {FP}, FN: {FN}")
print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, Accuracy: {accuracy:.2f}\n")

print("Gender Matching:")
print(f"  TP: {gTP}, FP: {gFP}, FN: {gFN}")
print(f"  Precision: {g_precision:.2f}, Recall: {g_recall:.2f}, F1-Score: {g_f1:.2f}, Accuracy: {g_accuracy:.2f}")

# Save
results_df.to_csv(r"C:\Users\aidan\Downloads\deity_matching_results_v7.csv", index=False, encoding="utf-8-sig")
