import pandas as pd
from fuzzywuzzy import process, fuzz
from sklearn.metrics import accuracy_score
import ast
import re
import matplotlib.pyplot as plt

# File paths
truth_path = r"C:\Users\aidan\Downloads\filtered_250_sampled_rows_v4.csv"
final_path = r"C:\Users\aidan\Downloads\3.final_full_cleaned_v3_no_cat.csv"

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

def save_df_as_png(df, filename, dpi=200, fontsize=10, row_height=0.6):
    nrows = len(df)

    # Rough estimate for initial figure width, will be corrected by auto_set_column_width
    fig_height = nrows * row_height + 1.5
    fig_width = 12  # Start with a moderate width

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)

    # Automatically adjust column widths based on content
    try:
        table.auto_set_column_width(col=list(range(len(df.columns))))
    except AttributeError:
        print("auto_set_column_width requires matplotlib >= 3.6")

    table.scale(1.0, 1.2)
    plt.tight_layout(pad=1.0)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close()

def parse_comma_column(column):
    if isinstance(column, str):
        stripped = column.strip()
        # Heuristic: looks like a list
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                evaluated = ast.literal_eval(stripped)
                if isinstance(evaluated, list):
                    return [normalize_name(item) for item in evaluated if normalize_name(item)]
            except (ValueError, SyntaxError):
                pass  # Fall through to comma split

        # Fallback: split by comma, strip and normalize each part, and filter out blanks
        return [normalize_name(part) for part in column.split(',') if normalize_name(part)]
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
# Build a mapping: alias (normalized) → base deity (normalized)
def build_alias_map(truth_df):
    alias_map = {}

    for _, row in truth_df.iterrows():
        deities = parse_comma_column(row.get("Deities", ""))
        aliases = parse_linked_column(row.get("Other_names", ""))

        for i, deity in enumerate(deities):
            if deity is None:
                continue

            # Include the base deity itself
            alias_map[deity] = deity

            # Handle corresponding alias(es)
            if i < len(aliases):
                for alias in parse_comma_column(aliases[i]):
                    if alias:
                        alias_map[alias] = deity
    return alias_map
def clean_final_deities_into_aliases(final_df, truth_df):
    for idx, row in final_df.iterrows():
        # Get corresponding truth row (assumes same index/order)
        truth_row = truth_df.iloc[idx]

        # Build per-row alias map
        alias_map = {}
        alias_keys = []

        base_deities = parse_comma_column(truth_row.get("Deities", ""))
        other_names = parse_linked_column(truth_row.get("Other_names", ""))

        for i, deity in enumerate(base_deities):
            if not deity:
                continue

            alias_map[deity] = deity
            alias_keys.append(deity)

            if i < len(other_names):
                for alias in parse_comma_column(other_names[i]):
                    if alias:
                        alias_map[alias] = deity
                        alias_keys.append(alias)

        # Parse predicted deities for this row
        raw_deities = parse_comma_column(row.get('deities', ''))
        cleaned_deities = []
        other_names_per_deity = {}

        for deity in raw_deities:
            if not deity:
                continue

            match, score = process.extractOne(deity, alias_keys, scorer=fuzz.token_sort_ratio)
            canonical = alias_map.get(match)

            if score >= 90 and canonical:
                if canonical not in cleaned_deities:
                    cleaned_deities.append(canonical)

                if canonical != deity:
                    if canonical not in other_names_per_deity:
                        other_names_per_deity[canonical] = []
                    other_names_per_deity[canonical].append(deity)
            else:
                cleaned_deities.append(deity)

        final_df.at[idx, 'deities'] = ", ".join(cleaned_deities)

        ordered_aliases = []
        for deity in cleaned_deities:
            aliases = other_names_per_deity.get(deity, [])
            ordered_aliases.append(", ".join(aliases))
        final_df.at[idx, 'other names'] = "; ".join(ordered_aliases)

    return final_df





# Evaluation function
def evaluate_row(uuid, truth_row, final_row):
    truth_deities = parse_comma_column(truth_row.get('Deities', ''))
    truth_genders = parse_comma_column(truth_row.get('Gender', ''))
    other_names = parse_linked_column(truth_row.get('Other_names', ''))

    final_deities = parse_comma_column(final_row.get('deities', ''))
    final_genders = parse_comma_column(final_row.get('gender', ''))
    other_names_final = parse_linked_column(final_row.get('other names', ''))
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

    # Identify false positives — unmatched predicted deities
    unmatched_predictions = [d for d in final_deities if d not in used_final_deities]
    score += [0 for _ in unmatched_predictions]  # Truth = 0, Predicted = 1
    gender_score += [0 for _ in unmatched_predictions]  # No gender match

    filtered_gender_score = [gs for ds, gs in zip(score, gender_score) if ds == 1]

    return {
        "uuid": uuid,
        "truth_deities": truth_row.get('Deities', ''),
        "truth_genders": truth_row.get('Gender', ''),
        "other_names": truth_row.get('Other_names', ''),
        "final_deities": final_row.get('deities', ''),
        "final_genders": final_row.get('gender', ''),
        "other_names_final": final_row.get('other names', ''),
        "matched_deities": str(matched_deities),
        "score": score,
        "gender_score": filtered_gender_score
    }



# Run evaluation
results = []
#alias_map = build_alias_map(truth_df)
#final_df = clean_final_deities_into_aliases(final_df, truth_df)

for _, truth_row in truth_df.iterrows():
    uuid = truth_row['uuid']
    final_row = final_df[final_df['uuid'] == uuid]
    if final_row.empty:
        continue
    result = evaluate_row(uuid, truth_row, final_row.iloc[0])
    results.append(result)
from collections import defaultdict

# Initialize counts
gender_match_stats = defaultdict(lambda: {"matched": 0, "unmatched": 0})

for row in results:
    truth_genders = parse_comma_column(row.get("truth_genders", ''))
    deity_scores = row.get("score", [])

    for gender, match in zip(truth_genders, deity_scores):
        gender_key = gender.strip().lower() if gender else "NA"
        if match == 1:
            gender_match_stats[gender_key]["matched"] += 1
        else:
            gender_match_stats[gender_key]["unmatched"] += 1

# Gender match breakdown
gender_match_stats = defaultdict(lambda: {"matched": 0, "unmatched": 0})

for row in results:
    truth_genders = parse_comma_column(row.get("truth_genders", ''))
    deity_scores = row.get("score", [])

    for gender, match in zip(truth_genders, deity_scores):
        gender_key = gender.strip().lower() if gender else "NA"
        if match == 1:
            gender_match_stats[gender_key]["matched"] += 1
        else:
            gender_match_stats[gender_key]["unmatched"] += 1

# Convert to pandas DataFrame
gender_stats_df = pd.DataFrame([
    {
        "Gender": gender,
        "Matched": counts["matched"],
        "Unmatched": counts["unmatched"],
        "Total": counts["matched"] + counts["unmatched"],
        "% Matched": round(100 * counts["matched"] / (counts["matched"] + counts["unmatched"]), 2)
        if (counts["matched"] + counts["unmatched"]) > 0 else 0.0
    }
    for gender, counts in gender_match_stats.items()
])
print(gender_stats_df)

# Optional: sort by total descending
gender_stats_df = gender_stats_df.sort_values(by="Total", ascending=False).reset_index(drop=True)

# Display the table
print(gender_stats_df)
save_df_as_png(gender_stats_df, r"C:\Users\aidan\Downloads\gender_eval_summary.png")

# Convert results
results_df = pd.DataFrame(results)

TP, FP, FN = 0, 0, 0
gTP, gFP, gFN = 0, 0, 0
matched_missing = 0

for row in results:
    # Safely parse matched_deities if read from string
    if isinstance(row["matched_deities"], str):
        matched_deities = ast.literal_eval(row["matched_deities"])
    else:
        matched_deities = row["matched_deities"]

    used_final_deities = set()

    for deity, match_info in matched_deities.items():
        if isinstance(match_info, dict):
            match = match_info["match"]
            score = match_info["score"]
        else:
            match, score = match_info  # fallback for older format

        if match == "missing" and str(row["truth_deities"]).strip().lower() == "missing":
            matched_missing += 1  # new counter
        elif score >= 50:
            TP += 1
            used_final_deities.add(match)
        else:
            if str(row["truth_deities"]).strip().lower() != "missing":
                FN += 1

    final_deities = parse_comma_column(row["final_deities"])
    unmatched_predictions = [d for d in final_deities if d not in used_final_deities]
    FP += len(unmatched_predictions)

    # Gender metrics
    gender_scores = row["gender_score"]
    deity_scores = row["score"]

    for ds, gs in zip(deity_scores, gender_scores):
        if ds == 1:
            if gs == 1:
                gTP += 1
            else:
                gFP += 1


# Compute metrics
precision = TP / (TP + FP) if (TP + FP) else 0
recall = TP / (TP + FN) if (TP + FN) else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
accuracy = TP / (TP + FP + FN) if (TP + FP + FN) else 0

g_precision = gTP / (gTP + gFP) if (gTP + gFP) else 0
g_recall = gTP / (gTP + gFN) if (gTP + gFN) else 0
g_f1 = 2 * g_precision * g_recall / (g_precision + g_recall) if (g_precision + g_recall) else 0
g_accuracy = gTP / (gTP + gFP + gFN) if (gTP + gFP + gFN) else 0


# Print results
print("Deity Matching:")
print(f"  TP: {TP}, FP: {FP}, FN: {FN}")
print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}, Accuracy: {accuracy:.2f}\n")

print("Gender Matching:")
print(f"  TP: {gTP}, FP: {gFP}, FN: {gFN}")
print(f"  Precision: {g_precision:.2f}, Recall: {g_recall:.2f}, F1-Score: {g_f1:.2f}, Accuracy: {g_accuracy:.2f}")

print(f"  Matched Missing: {matched_missing}")
# Collect and save false positives and false negatives
false_positives = []
false_negatives = []

for row in results:
    if isinstance(row["matched_deities"], str):
        matched_deities = ast.literal_eval(row["matched_deities"])
    else:
        matched_deities = row["matched_deities"]

    used_final_deities = set()
    for deity, match_info in matched_deities.items():
        if isinstance(match_info, dict):
            match = match_info["match"]
            score = match_info["score"]
        else:
            match, score = match_info

        if score >= 50:
            used_final_deities.add(match)
        else:
            if str(row["truth_deities"]).strip().lower() == "missing":
                continue  # Skip FN if truth_deities is 'missing'
            # False Negative
            false_negatives.append({
                "uuid": row["uuid"],
                "missing_deity": deity,
                "truth_deities": row["truth_deities"],
                "predicted_deities": row["final_deities"],
                "matched_deities": row["matched_deities"]
            })

    final_deities = parse_comma_column(row["final_deities"])
    for d in final_deities:
        if d not in used_final_deities:
            # False Positive
            false_positives.append({
                "uuid": row["uuid"],
                "extra_deity": d,
                "truth_deities": row["truth_deities"],
                "predicted_deities": row["final_deities"],
                "matched_deities": row["matched_deities"]
            })

# Save FP and FN to CSV
fp_df = pd.DataFrame(false_positives)
fn_df = pd.DataFrame(false_negatives)

fp_df.to_csv(r"C:\Users\aidan\Downloads\deity_false_positives_v3.csv", index=False, encoding="utf-8-sig")
fn_df.to_csv(r"C:\Users\aidan\Downloads\deity_false_negatives_v3.csv", index=False, encoding="utf-8-sig")

# Save
results_df = results_df.merge(truth_df[['uuid', 'ambiguous']], on='uuid', how='left')
results_df.to_csv(r"C:\Users\aidan\Downloads\deity_matching_results_full_v4_no_cat.csv", index=False, encoding="utf-8-sig")

import pandas as pd
import ast

def evaluate_deities(truth_df, final_df):
    #alias_map = build_alias_map(truth_df)
    #final_df = clean_final_deities_into_aliases(final_df, truth_df)

    results = []

    total_truth_deities = 0
    total_gpt_deities = 0

    for _, truth_row in truth_df.iterrows():
        uuid = truth_row['uuid']
        final_row = final_df[final_df['uuid'] == uuid]
        if final_row.empty:
            continue
        result = evaluate_row(uuid, truth_row, final_row.iloc[0])
        results.append(result)

        # Count deities from truth and final
        truth_deities = parse_comma_column(truth_row.get('Deities', ''))
        final_deities = parse_comma_column(final_row.iloc[0].get('deities', ''))

        total_truth_deities += sum(1 for d in truth_deities if d)
        normalized_deities = set(normalize_name(d) for d in final_deities if d)
        total_gpt_deities += len(normalized_deities)

    TP, FP, FN = 0, 0, 0
    matched_missing = 0
    for row in results:
        matched_deities = ast.literal_eval(row["matched_deities"]) if isinstance(row["matched_deities"], str) else row["matched_deities"]
        used_final_deities = set()

        for deity, match_info in matched_deities.items():
            if isinstance(match_info, dict):
                match = match_info["match"]
                score = match_info["score"]
            else:
                match, score = match_info

            if match == "missing" and str(row["truth_deities"]).strip().lower() == "missing":
                matched_missing += 1  # new counter
            elif score >= 50:
                TP += 1
                used_final_deities.add(match)
            else:
                if str(row["truth_deities"]).strip().lower() != "missing":
                    FN += 1

        final_deities = parse_comma_column(row["final_deities"])
        unmatched_predictions = [d for d in final_deities if d not in used_final_deities]
        FP += len(unmatched_predictions)

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) else 0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Matched Missing": matched_missing,
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1": round(f1, 2),
        "Accuracy": round(accuracy, 2),
        "Truth_Deities": total_truth_deities,
        "GPT_Deities": total_gpt_deities
    }

# Load all combinations
truth = pd.read_csv(r"C:\Users\aidan\Downloads\filtered_250_sampled_rows.csv")
truth_v2 = pd.read_csv(r"C:\Users\aidan\Downloads\filtered_250_sampled_rows_v2.csv")
truth_v3 = pd.read_csv(r"C:\Users\aidan\Downloads\filtered_250_sampled_rows_v3.csv")
truth_v4 = pd.read_csv(r"C:\Users\aidan\Downloads\filtered_250_sampled_rows_v4.csv")
final = pd.read_csv(r"C:\Users\aidan\Downloads\3.final_full_cleaned.csv")
final_v2 = pd.read_csv(r"C:\Users\aidan\Downloads\3.final_full_cleaned_v2.csv")
final_v3 = pd.read_csv(r"C:\Users\aidan\Downloads\3.final_full_cleaned_v3.csv")
final_v3_out_info = pd.read_csv(r"C:\Users\aidan\Downloads\3.final_full_cleaned_v3_out_info.csv")
final_v3_no_cat = pd.read_csv(r"C:\Users\aidan\Downloads\3.final_full_cleaned_v3_no_cat.csv")



# Prepare comparison matrix
combinations = [
    ("Truth 1 with Prompt 1", truth, final),
    ("Truth 1 vs Prompt 2", truth, final_v2),
    ("Truth 2 vs Prompt 1", truth_v2, final),
    ("Truth 2 vs Prompt 2", truth_v2, final_v2),
    ("Truth 3 vs Prompt 2", truth_v3, final_v2), 
    ("Truth 4 vs Prompt 3", truth_v4, final_v3), 
    ("Truth 4 vs Prompt 3 with outside info column", truth_v4, final_v3_out_info), 
    ("Truth 4 vs Prompt 3 without categories", truth_v4, final_v3_no_cat), 
]

results_summary = []

for name, t_df, f_df in combinations:
    metrics = evaluate_deities(t_df, f_df)
    metrics["Comparison"] = name
    results_summary.append(metrics)

# Create and display the summary table
summary_df = pd.DataFrame(results_summary)
summary_df = summary_df[["Comparison", "TP", "FP", "FN", "Matched Missing", "Precision", "Recall", "F1", "Accuracy", "Truth_Deities", "GPT_Deities"]]
print(summary_df)

# Save with improved layout
save_df_as_png(summary_df, r"C:\Users\aidan\Downloads\deity_eval_summary.png")


ambiguous_1_df = results_df[results_df["ambiguous"] == 1]
ambiguous_0_df = results_df[results_df["ambiguous"] == 0]

def evaluate_subset(results_subset):
    TP, FP, FN = 0, 0, 0
    matched_missing = 0
    total_truth_deities = 0
    total_gpt_deities = 0

    for _, row in results_subset.iterrows():
        matched_deities = ast.literal_eval(row["matched_deities"]) if isinstance(row["matched_deities"], str) else row["matched_deities"]
        used_final_deities = set()

        for deity, match_info in matched_deities.items():
            if isinstance(match_info, dict):
                match = match_info["match"]
                score = match_info["score"]
            else:
                match, score = match_info

            if match == "missing" and str(row["truth_deities"]).strip().lower() == "missing":
                matched_missing += 1
            elif score >= 50:
                TP += 1
                used_final_deities.add(match)
            else:
                if str(row["truth_deities"]).strip().lower() != "missing":
                    FN += 1

        final_deities = parse_comma_column(row["final_deities"])
        unmatched_predictions = [d for d in final_deities if d not in used_final_deities]
        FP += len(unmatched_predictions)

        # Count raw totals
        truth_deities = parse_comma_column(row.get("truth_deities", ''))
        total_truth_deities += sum(1 for d in truth_deities if d)
        total_gpt_deities += sum(1 for d in final_deities if d)

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    accuracy = TP / (TP + FP + FN) if (TP + FP + FN) else 0

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Matched Missing": matched_missing,
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1": round(f1, 2),
        "Accuracy": round(accuracy, 2),
        "Truth_Deities": total_truth_deities,
        "GPT_Deities": total_gpt_deities
    }
# Evaluate both groups
amb_1_metrics = evaluate_subset(ambiguous_1_df)
amb_1_metrics["Comparison"] = "Ambiguous = 1"

amb_0_metrics = evaluate_subset(ambiguous_0_df)
amb_0_metrics["Comparison"] = "Ambiguous = 0"

# Combine and display
amb_summary_df = pd.DataFrame([amb_0_metrics, amb_1_metrics])
amb_summary_df = amb_summary_df[["Comparison", "TP", "FP", "FN", "Matched Missing", "Precision", "Recall", "F1", "Accuracy", "Truth_Deities", "GPT_Deities"]]
print(amb_summary_df)

# Optional: save
save_df_as_png(amb_summary_df, r"C:\Users\aidan\Downloads\deity_eval_ambiguous_summary.png")
