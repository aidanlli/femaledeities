import pandas as pd
import re

output_path = "C:/Users/aidan/Downloads/updated_output_12_subjects.csv"

df = pd.read_csv(output_path)

# Define keywords with each pronoun as a separate keyword
male_keywords = ["he", "him", "his", "father", "king", "lord", "prince", 
                 "son", "husband", "brother", "patriarch", "fatherhood", "brotherhood"]
female_keywords = ["she", "her", "mother", "queen", "lady", "goddess", 
                   "daughter", "wife", "sister", "matriarch", "motherhood"]
dual_keywords = ["androgynous", "dual-gendered", "god/goddess"]

# Function to count keywords
def count_keywords(column, keywords):
    counts = {}
    total = 0
    for keyword in keywords:
        # Count each keyword individually, case insensitive
        count = column.str.count(rf'\b{re.escape(keyword)}\b', flags=re.IGNORECASE).sum()
        counts[keyword] = count
        total += count  # Add to overall count for the category
    return counts, total

# Count keywords in the Text column of the updated DataFrame
male_counts, male_total = count_keywords(df['Text'], male_keywords)
female_counts, female_total = count_keywords(df['Text'], female_keywords)
dual_counts, dual_total = count_keywords(df['Text'], dual_keywords)

# Display individual and overall counts
print("Male Keywords Count:")
for k, v in male_counts.items():
    print(f"  {k}: {v}")
print(f"Total Male Keywords: {male_total}")

print("\nFemale Keywords Count:")
for k, v in female_counts.items():
    print(f"  {k}: {v}")
print(f"Total Female Keywords: {female_total}")

print("\nDual Keywords Count:")
for k, v in dual_counts.items():
    print(f"  {k}: {v}")
print(f"Total Dual Keywords: {dual_total}")
