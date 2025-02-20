import pandas as pd
import re
import matplotlib.pyplot as plt

output_path = "C:/Users/aidan/Downloads/updated_output_12_subjects.csv"
export_dir = "C:/Users/aidan/OneDrive/Documents/GitHub/femaledeities/plots_and_stats/"  # Ensure trailing slash

df = pd.read_csv(output_path)

# Define keywords
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
        count = column.str.count(rf'\b{re.escape(keyword)}\b', flags=re.IGNORECASE).sum()
        counts[keyword] = count
        total += count
    return counts, total

# Count keywords
male_counts, male_total = count_keywords(df['Text'], male_keywords)
female_counts, female_total = count_keywords(df['Text'], female_keywords)
dual_counts, dual_total = count_keywords(df['Text'], dual_keywords)

# Convert keyword counts to DataFrames
male_df = pd.DataFrame(list(male_counts.items()), columns=["Male Keyword", "Count"])
male_df.loc[len(male_df)] = ["Total Male Keywords", male_total]

female_df = pd.DataFrame(list(female_counts.items()), columns=["Female Keyword", "Count"])
female_df.loc[len(female_df)] = ["Total Female Keywords", female_total]

dual_df = pd.DataFrame(list(dual_counts.items()), columns=["Dual Keyword", "Count"])
dual_df.loc[len(dual_df)] = ["Total Dual Keywords", dual_total]

# Function to save a DataFrame as an image
def save_df_as_image(df, filename, title):
    fig, ax = plt.subplots(figsize=(6, len(df) * 0.5 + 1))  # Adjust figure size based on rows
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    plt.title(title, fontsize=12, fontweight="bold")
    plt.savefig(f"{export_dir}{filename}", bbox_inches="tight", dpi=300)
    plt.close()

# Function to save a bar chart
def save_bar_chart(df, filename, title, x_label, y_label):
    plt.figure(figsize=(10, 5))
    plt.bar(df.iloc[:-1, 0], df.iloc[:-1, 1], color='skyblue')  # Exclude the total row
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{export_dir}{filename}", dpi=300)
    plt.close()

# Save keyword tables and charts
save_df_as_image(male_df, "male_keywords.png", "Male Keywords Count")
save_bar_chart(male_df, "male_keywords_bar.png", "Male Keywords Frequency", "Keyword", "Count")

save_df_as_image(female_df, "female_keywords.png", "Female Keywords Count")
save_bar_chart(female_df, "female_keywords_bar.png", "Female Keywords Frequency", "Keyword", "Count")

save_df_as_image(dual_df, "dual_keywords.png", "Dual Keywords Count")
save_bar_chart(dual_df, "dual_keywords_bar.png", "Dual Keywords Frequency", "Keyword", "Count")

# Count unique values for categorical columns
columns_to_count = ["DocType", "Culture", "Region", "Subregion", "Subsistence"]

for column in columns_to_count:
    count_df = df[column].value_counts().reset_index()
    count_df.columns = [column, "Count"]
    total_row = pd.DataFrame({column: [f"Total {column}"], "Count": [count_df["Count"].sum()]})
    count_df = pd.concat([count_df, total_row], ignore_index=True)
        
    save_df_as_image(count_df, f"{column}_counts.png", f"{column} Value Counts")
    if column != "Culture":
        save_bar_chart(count_df.iloc[:-1], f"{column}_counts_bar.png", f"{column} Frequency", column, "Count")  # Exclude total row
        
# Special case: Histogram for Culture
plt.figure(figsize=(10, 5))
df["Culture"].value_counts().plot(kind='hist', bins=50, color='lightcoral', edgecolor='black')
plt.xlabel("Culture")
plt.ylabel("Frequency")
plt.title("Culture Count Distribution")
plt.tight_layout()
plt.savefig(f"{export_dir}Culture_histogram.png", dpi=300)
plt.close()

print("Tables and charts saved as PNG images in:", export_dir)
