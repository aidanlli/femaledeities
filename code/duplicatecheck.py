import pandas as pd

# File paths
csv_path = r"C:/Users/aidan/Downloads/qrySummary_eHRAF_WorldCultures_Jan2024.csv"
concatenated_csv_path = r"C:/Users/aidan/Downloads/concatenated_output.csv"

# Load the data
df_summary = pd.read_csv(csv_path)
df_concatenated = pd.read_csv(concatenated_csv_path)

# Extract unique culture names (strip spaces for consistency)
summary_cultures = set(df_summary["EHRAF WORLD CULTURES NAME"].dropna().str.strip())
concatenated_cultures = set(df_concatenated["Culture"].dropna().str.strip())

# Find cultures in qrySummary but NOT in concatenated_output
missing_in_concatenated = summary_cultures - concatenated_cultures

# Save the missing cultures to a CSV file
missing_cultures_path = r"C:/Users/aidan/Downloads/missing_cultures.csv"
pd.DataFrame({"Missing Cultures": list(missing_in_concatenated)}).to_csv(missing_cultures_path, index=False)

# Print the result - we should see the following: ['Eastern Apache', 'Baluchi', 'Karakalpak', 'Hazara', 'Ghorbat', 'Tajiks']
print(f"Total missing cultures: {len(missing_in_concatenated)}")
print(f"Missing cultures saved to: {missing_cultures_path}")
print(list(missing_in_concatenated))
