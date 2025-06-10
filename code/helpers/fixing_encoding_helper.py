import pandas as pd
# In case the encoding of the 250 sampled rows file is incorrect
# File paths
processed_file = "C:/Users/aidan/Downloads/processed_output_11s_final.csv"
sampled_file = "C:/Users/aidan/Downloads/250_sampled_rows_truth.csv"
output_file = "C:/Users/aidan/Downloads/250_sampled_rows_encoding.csv"

# Load the data
processed_df = pd.read_csv(processed_file, encoding="utf-8-sig")
sampled_df = pd.read_csv(sampled_file, encoding="utf-8-sig")

# Extract the uuids from sampled data
uuids_to_match = set(sampled_df["uuid"])

# Filter the processed data to match uuids
matched_rows = processed_df[processed_df["uuid"].isin(uuids_to_match)]

# Save the filtered rows to a new CSV
matched_rows.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Saved {len(matched_rows)} matched rows to {output_file}")
