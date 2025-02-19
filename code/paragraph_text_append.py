import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import os

# File paths
input_path = "C:/Users/aidan/Downloads/concatenated_output_12s.csv"
output_path = "C:/Users/aidan/Downloads/updated_output_12_subjects.csv"
save_interval = 1000  # Save every 500 rows

# Load input CSV
df = pd.read_csv(input_path)

# Try loading existing output CSV to resume progress
if os.path.exists(output_path):
    print("Resuming from existing file...")
    df_existing = pd.read_csv(output_path)
    for col in ["Raw Text", "Text", "Image Link"]:
        if col in df_existing.columns:
            df[col] = df_existing[col]  # Preserve already scraped data
        else:
            df[col] = None  # Initialize missing columns
else:
    df["Raw Text"] = None
    df["Text"] = None
    df["Image Link"] = None

# User-Agent Header
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

session = requests.Session()

# Function to scrape text and image link
def scrape_text(url):
    try:
        with  session.get(url, headers=headers, timeout=10) as response:

          response.raise_for_status()
          soup = BeautifulSoup(response.text, "html.parser")
  
          # :one: First: Check for <span> with an id and store in Raw Text
          span_with_id = soup.find("span", id=True)
          raw_text = span_with_id.get_text(strip=True) if span_with_id else "No text found"
  
          # :two: Check for <span> with hraf-doc__sre--content class
          hraf_span = soup.find("span", class_="hraf-doc__sre--content")
          if hraf_span:
              return raw_text, hraf_span.get_text(strip=True), None
  
          # :three: Check for <p> with specific class
          paragraph = soup.find("p", class_="mdc-typography--body2 undefined")
          if paragraph:
              return raw_text, paragraph.get_text(strip=True), None
  
          # :four: If no <p>, check inside <span>
          span = soup.find("span", class_="mdc-typography--body2 linegroup")
          if span:
              return raw_text, span.get_text(strip=True), None
  
          # :five: If no <span>, check <figcaption>
          figcaption = soup.find("figcaption")
          if figcaption:
              image_link = None
              img_tag = figcaption.find_previous("a", class_="img")
              if img_tag and "href" in img_tag.attrs:
                  image_link = img_tag["href"]  # Extract image link
              return raw_text, figcaption.get_text(strip=True), image_link

          return raw_text, "No text found", None
    
    except requests.exceptions.HTTPError as e:
        if response.status_code == 451:
            return "Unavailable due to legal reasons", None, None
        return f"Error: {e}", None, None
    except requests.RequestException as e:
        return f"Error: {e}", None, None
        
# Scrape only missing rows
tqdm.pandas()
for i, row in tqdm(df.iterrows(), total=len(df)):
    if pd.isna(row["Text"]):  # Only scrape if Text column is empty
        raw_text, text, image_link = scrape_text(row["Permalink"])
        df.at[i, "Raw Text"] = raw_text
        df.at[i, "Text"] = text
        df.at[i, "Image Link"] = 'Image:' + image_link if image_link else None

    # Save progress every N rows
    if i % save_interval == 0:
        df.to_csv(output_path, index=False)

# Final save
df.to_csv(output_path, index=False)
print(f"Updated CSV saved to {output_path}")

