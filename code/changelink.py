import pandas as pd

# Load both CSVs
df_pn = pd.read_csv("C:/Users/aidan/Downloads/complete_deity_text_dataframe_pn.csv" , encoding="utf-8-sig")
df = pd.read_csv("C:/Users/aidan/Downloads/complete_deity_text_dataframe.csv" , encoding="utf-8-sig")

# Merge to bring in the replacement Text column
df_merged = df_pn.merge(df[['uuid', 'Text']], on='uuid', how='left', suffixes=('', '_new'))

# Replace the original Text with the new one
df_merged['Text'] = df_merged['Text_new']

# Drop the temporary merged column
df_merged = df_merged.drop(columns=['Text_new'])


# Save to CSV
df_merged.to_csv("C:/Users/aidan/Downloads/complete_deity_text_dataframe_pn.csv", index=False, encoding="utf-8-sig")
