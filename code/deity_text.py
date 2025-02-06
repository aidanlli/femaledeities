import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os
# File paths
input_path = "C:/Users/aidan/Downloads/concatenated_output.csv"
output_path = "C:/Users/aidan/Downloads/updated_output_test.csv"
save_interval = 100  # Save every 100 rows

# Load input CSV
df = pd.read_csv(input_path)

# Try loading existing output CSV to resume progress
if os.path.exists(output_path):
    print("Resuming from existing file...")
    df_existing = pd.read_csv(output_path)
    for col in ["Text", "Image Link"]:
        if col in df_existing.columns:
            df[col] = df_existing[col]  # Preserve already scraped data
        else:
            df[col] = None  # Initialize missing columns
else:
    df["Text"] = None
    df["Image Link"] = None

session = requests.Session()

# Function to scrape text and image link
def scrape_text(url):
    try:
        response = session.get(url, timeout=10)  
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # 1️⃣ Try extracting text from <p>
        paragraph = soup.find("p", class_="mdc-typography--body2 undefined")
        if paragraph:
            return paragraph.get_text(strip=True), None

        # 2️⃣ If no <p>, check inside <span>
        span = soup.find("span", class_="mdc-typography--body2 linegroup")
        if span:
            return span.get_text(strip=True), None

        # 3️⃣ If no <span>, check <figcaption>
        figcaption = soup.find("figcaption")
        if figcaption:
            image_link = None

            # Look for <a> tag with class "img"
            img_tag = figcaption.find_previous("a", class_="img")
            if img_tag and "href" in img_tag.attrs:
                image_link = img_tag["href"]  # Extract image link

            return figcaption.get_text(strip=True), image_link

        return "No text found", None

    except requests.RequestException as e:
        return f"Error: {e}", None

# Scrape only missing rows
tqdm.pandas()
for i, row in tqdm(df.iterrows(), total=len(df)):
    if pd.isna(row["Text"]):  # Only scrape if Text column is empty
        text, image_link = scrape_text(row["Permalink"])
        df.at[i, "Text"] = text
        df.at[i, "Image Link"] = 'Image:' + image_link if image_link else None  # Save image link

    # Save progress every N rows
    if i % save_interval == 0:
        df.to_csv(output_path, index=False)

# Final save
df.to_csv(output_path, index=False)
print(f"Updated CSV saved to {output_path}")
