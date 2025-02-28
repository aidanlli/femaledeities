import pandas as pd
import matplotlib.pyplot as plt

# File paths
input_csv = "C:/Users/aidan/Downloads/processed_output_11s.csv"
output_csv = "C:/Users/aidan/Downloads/processed_output_11s.csv"
output_image_path = "C:/Users/aidan/OneDrive/Documents/GitHub/femaledeities/plots_and_stats/topic_counts.png"
output_bar_chart_path = "C:/Users/aidan/OneDrive/Documents/GitHub/femaledeities/plots_and_stats/topic_counts_bar_chart.png"

# Load CSV
df = pd.read_csv(input_csv)
df["Text"] = df["Text"].apply(lambda x: x.split("Load in Context", 1)[-1] if isinstance(x, str) and "Load in Context" in x else x)

# Define topics with keys as the topic numbers (as strings)
topics = {
    "769": "769 - Cult of the Dead",
    "771": "771 - General Character of Religion",
    "772": "772 - Cosmology",
    "773": "773 - Mythology",
    "774": "774 - Animism",
    "775": "775 - Eschatology",
    "776": "776 - Spirits and Gods",
    "777": "777 - Luck and Chance",
    "778": "778 - Sacred Objects and Places",
    "779": "779 - Theological Systems",
    "787": "787 - Revelation and Divination"
}

# Helper function: check if topic_number occurs after "Load in Context"
def topic_in_text(raw_text, topic_number):
    if not isinstance(raw_text, str):
        return 0
    # Find the position of "Load in Context"
    pos = raw_text.find("Load in context")
    if pos == -1:
        return 0
    # Only consider the text after (and including) "Load in Context"
    substring = raw_text[pos:]
    return 1 if topic_number in substring else 0

# Create new columns for each topic by checking in 'Raw Text'
for key, col_name in topics.items():
    df[col_name] = df["Raw Text"].apply(lambda x: topic_in_text(x, key))
    print(f"Processed topic column: {col_name}")

# Optionally, save the modified CSV
df.to_csv(output_csv, index=False)
print(f"Modified CSV saved to {output_csv}")

# Now count the occurrences (i.e. sum the 1's) in each topic column
topic_columns = list(topics.values())
topic_counts = df[topic_columns].sum().reset_index()
topic_counts.columns = ["Topic", "Count"]

# Create a table plot using matplotlib
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis("tight")
ax.axis("off")
table = ax.table(cellText=topic_counts.values,
                 colLabels=topic_counts.columns,
                 cellLoc="center", loc="center")

# Save the table as a PNG image in the export folder
plt.savefig(output_image_path, bbox_inches="tight", dpi=300)
print(f"Topic count table saved as {output_image_path}")

# Create a bar chart for the topic counts
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.bar(topic_counts["Topic"], topic_counts["Count"], color='skyblue')
ax2.set_xlabel("Topic")
ax2.set_ylabel("Count")
ax2.set_title("Counts of Subjects")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

# Save the bar chart as a PNG image in the export folder
plt.savefig(output_bar_chart_path, bbox_inches="tight", dpi=300)
print(f"Topic count bar chart saved as {output_bar_chart_path}")
