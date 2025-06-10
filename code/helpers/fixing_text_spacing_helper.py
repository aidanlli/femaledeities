import pandas as pd
import re

# Load the file
file_path = "C:/Users/aidan/Downloads/complete_deity_text_dataframe_pn.csv"
df = pd.read_csv(file_path, encoding="utf-8-sig")

def clean_text(text):
    if pd.isna(text):
        return text

    # Remove newline characters within paragraphs
    text = re.sub(r'\n+', ' ', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s{2,}', ' ', text)

    # Add space after punctuation if missing
    text = re.sub(r'([.,;:!?])([^\s])', r'\1 \2', text)

    # Optionally: separate concatenated words with a lowercase followed by uppercase
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    return text.strip()

# Apply the cleaning function
df['Text'] = df['Text'].apply(clean_text)

# Optionally save the cleaned dataframe
df.to_csv("C:/Users/aidan/Downloads/complete_deity_text_dataframe_pn_test.csv", index=False, encoding="utf-8-sig")
