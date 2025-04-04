import pandas as pd
import spacy
from tqdm import tqdm  # Import tqdm for progress bar

# Load the SpaCy English model (use 'en_core_web_sm', 'en_core_web_md', or 'en_core_web_lg')
nlp = spacy.load("en_core_web_lg")

# Load CSV file
file_path = "C:/Users/aidan/Downloads/complete_deity_text_dataframe.csv"
output_file = "C:/Users/aidan/Downloads/complete_deity_text_dataframe_pn.csv"

df = pd.read_csv(file_path, encoding = "utf-8-sig")

# Initialize tqdm to track progress
tqdm.pandas()

# Function to extract proper nouns with correct SpaCy POS tag
def extract_proper_nouns(text):
    doc = nlp(str(text))  # Convert to string in case of NaNs
    proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]  # Corrected POS tag
    return proper_nouns

# Check if the 'Proper_Nouns' column exists; if not, create it
if "Proper_Nouns" not in df.columns:
    df["Proper_Nouns"] = None  # Initialize empty column

batch_size = 5000
mask = df["Proper_Nouns"].isna()

for i in tqdm(range(len(df))):
    if mask[i]:  # Only process rows that haven't been processed yet
        df.at[i, "Proper_Nouns"] = extract_proper_nouns(df.at[i, "Text"])

    # Periodically save every `batch_size` rows
    if i % batch_size == 0 and i > 0:
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"Progress saved at row {i}")


# Save the updated CSV with proper nouns column
df.to_csv(output_file, index=False, encoding = "utf-8-sig")

print("Processing complete! Output saved to:", output_file)

# Display first few rows
df.head()
