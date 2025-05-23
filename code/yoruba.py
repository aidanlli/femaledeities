import pandas as pd

# File paths
full_file_path = "C:/Users/aidan/Downloads/complete_deity_text_dataframe_pn_test.csv"
sampled_uuids_path = "C:/Users/aidan/Downloads/250_sampled_rows_truth.csv"
output_file_path = "C:/Users/aidan/Downloads/filtered_250_sampled_rows.csv"

# Load full dataframe and sampled UUIDs
df_full = pd.read_csv(full_file_path, encoding="utf-8-sig")
df_sampled = pd.read_csv(sampled_uuids_path, encoding="utf-8-sig")

# Ensure 'uuid' column exists in both
if 'uuid' not in df_full.columns or 'uuid' not in df_sampled.columns:
    raise ValueError("Missing 'uuid' column in one of the files.")

# Merge while preserving the order of df_sampled
filtered_df = df_sampled[['uuid']].merge(df_full, on='uuid', how='left')

# Save the ordered filtered dataframe
filtered_df.to_csv(output_file_path, index=False, encoding="utf-8-sig")
print(f"Filtered CSV saved to: {output_file_path}")
