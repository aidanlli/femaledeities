import pandas as pd

# Define the output path
output_path = "C:/Users/aidan/Downloads/processed_output_11s.csv"
output_path_2 = "C:/Users/aidan/Downloads/processed_output_11s_updated.csv"


# Load the CSV file into a DataFrame
df = pd.read_csv(output_path)

def fix_encoding(text):
    if isinstance(text, str):  # Ensure it's a string before processing
        try:
            return text.encode("latin1").decode("utf-8")
        except UnicodeEncodeError:
            return text  # If it fails, return original text
    return text

# Apply to both "Raw Text" and "Text" columns
df["Raw Text"] = df["Raw Text"].apply(fix_encoding)
df["Text"] = df["Text"].apply(fix_encoding)

# Save the fixed CSV
df.to_csv(output_path_2, index=False, encoding="utf-8-sig")
print(f"Encoding fixed and saved to {output_path_2}")



df2 = pd.read_csv(output_path_2)

# Define the UUID to search for
uuid_to_find = "fa5057ed-1b89-4f3c-8156-0a2eb363e6be"

# Find the row where the "uuid" column matches the given value
matching_row = df2[df2["uuid"] == uuid_to_find]

# Extract the value from the "Text" column
if not matching_row.empty:
    text_value = matching_row["Text"].values[0]
    print("Text value:", text_value)
else:
    print("UUID not found in the file.")
