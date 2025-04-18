# === SETUP ===
import openai
import pandas as pd
from tqdm import tqdm
import time
import os
import json

# === SETUP ===
openai.api_key = ""
input_csv_path = "C:/Users/aidan/Downloads/filtered_250_sampled_rows.csv"
output_csv_path = "C:/Users/aidan/Downloads/3.final.csv"
batch_size = 2
temperature = 0

# === LOAD DATA ===
df = pd.read_csv(input_csv_path, encoding="utf-8-sig")
if os.path.exists(output_csv_path):
    result_df = pd.read_csv(output_csv_path, encoding="utf-8-sig")
    last_processed_index = result_df.index[-1]
else:
    result_df = pd.DataFrame()
    last_processed_index = -1

# === JSON PARSER ===
def extract_structured_data(text):
    try:
        # Remove code block markdown if present
        if text.startswith("```json"):
            text = text.strip("```json").strip("` \n")
        elif text.startswith("```"):
            text = text.strip("```").strip("` \n")
        
        data = json.loads(text)
        return pd.DataFrame([data])
    except Exception as e:
        print(f"[!] JSON parsing failed: {e}")
        return pd.DataFrame([{"raw_response": text, "error": str(e)}])

# === SYSTEM PROMPT ===
system_prompt = "Follow instructions precisely as if you were a renowned anthropologist and an economist professor is to determine the following tasks."

# === MAIN LOOP ===
for i in tqdm(range(last_processed_index + 1, len(df), batch_size), desc="Processing"):
    batch = df.iloc[i:i + batch_size]

    for idx, row in batch.iterrows():
        paragraph = row['Text']
        user_prompt = f"""
Using the paragraph below, follow steps 1 to 8.

Paragraph: {paragraph}

For the above paragraph; with temperature = 0, from each paragraph in column ""Text"", using GPT to read and interpret each paragraph like a human would (using full language understanding, not pattern-matching), please follow these steps:

1. For the paragraph, identify, if any, all the supernatural beings, including deities, divine or semi-divine beings, ancestors, ghosts, spirits, totems, or even mediums (priests, nuns, prophets, ...). 
When you do the identification, limit your answer to what you can infer from the information of the paragraph, do not use external sources. 

2. For each identified being, please estimate if they fit into any of the following categories, using as a general guide the list of suggested keywords, and using your understanding of the paragraph only. Do that for each category below (from 2.1 to 2.20)
2.1. Creator of the universe (“creator”, “founder”, “father”, “mother”, “created”, “conceived”) of/the (“cosmos”, “universe”, “world”, “earth”), 
2.2. Creator of humankind (“creator”, “founder”, “father”, “mother”, “mould”, “created”) of/the (“people”, “race”, “human”, “mankind”, “our mother” or “our father”, “made people”), 
2.3. Mother (of the cosmos, humankind, other gods or specific; “mother”, “first seed”, “cosmic egg”, “womb”,  “her children”, “birthed”, “fertility” or Creator of humankind AND female),
2.4. Wife (Wife of another deity or super natural or semi-divine being), 
2.5. Primal (“primal”, “oldest”, “prime”, “foundation”), 
2.6. Omnipresent or Omnipotent (“all-wise”, “all-powerful”, “all things”, “all people”),
2.7. Present (“intermediary”, “invocation”, “pay him homage”, “patron of"",  “possesses people”),
2.8. Absent (“from afar”, “not intervening”, “distant”, “absence”, “contemplation”, “indifferent”, “watches from”),
2.9. Warrior or hero (“aar”, “hero”, “military”, “battlefield”, “blade”, “blow”, “spear"", “spear”, “blood”, “conquest”, “enemies”, “battle”),
2.10. Related to nature (“fertility”, “seasons”, “water”, “animal”, “river”) 
2.11. Related to cosmos (“sky”, “storm”, “stars”, “sun”, “moon”, “clouds”, “time”), 
2.12. Related to death (“death”, “underworld”, “dead”, “afterlife”, “judgement”, “souls”, “fate”, “ghost”) 
2.13.Supreme ruler (“law”, “order”, “sovereign” , “kingship”); 
2.14. Dualistic (“opposites”, “twin”, “embrace” , [opposite words in the same sentence], “neither could exist without the other”),
2.15. Trickster (""mischievous"", “chaos”, “deceiver”, “deceit”), 
2.16. Evil (“demon”, “tyranny”, “vengeance”, “devil”, “evil”, “malevolent”, “destructive”), 
2.17. Good (“benevolent”, “mercy”, “nurturing”, “mediator”, “compassion”, “protection, “healing”, “welfare”, “redemption”, “salvation”),
2.18. Demigods (“sacred king”, “ancestor”, “supernatural”, “elf”, “elves”, “ghost”, “embodiment of the god”, “minor divinity”, “giant”, “apparitions”, “spirit”, “nommo”), 
2.19. Intermediaries (“shaman”, “shamanka”, “priest”, “priestess”, “nun”, “diviner”, “angel”, “saint”, “prophet”, “mystic”, “monk”, “oracle””),
2.20. Other - specify (use this column if there is an additional categorization that is obvious from the paragraph and it is not included in 2.1 to 2.19.

3. For each identified being or group of beings, please infer its or their gender from the paragraph:  
3.1. For each individual being, classify into: ""male"", ""female"", ""androgenous"" if it is a single being with both male & female genders, or ""genderless"" if it does not have any gender,
3.2. For each groups of beings, classify into: ""male"" if all members are male, ""female"" if all members are female, or  ""general"" in case of a category of beings where members have different genders.
3.3. If you can't infer the gender or genders from the paragraph, classify it as ""missing"".  

4. For every deity identified (and when none is), also add a degree of certainty that is a super natural being (or that there is none), based on how sure you are of your answer, using a continuous scale from 0 to 100, where 0 is that you don't know and 100 is that you are 100% sure.

5. For every gender assigned, also add a degree of certainty that is a super natural being based on how sure you are of your answer, using a continuous scale from 0 to 100, where 0 is that you don't know and 100 is that you are 100% sure.

6. Oganize this information in the following way: add to the table the following columns, creating a file called 3.final.csv; 
6.1. In a column called ""deities"" create a list of strings of all supernatural beings identified in (1), each of them in quotes (ie. ""deity1"") and separated by a comma (ie. ""deity1"", ""deity2"", ""deity3"")
6.2. In another column called ""gender"", in the same order as the list in (6.1) list the genders identified in (3) as strings (ie. male, female, missing);
6.3. Add two more columns called ""certainty_deity"" and ""certainty_gender"" and list, separated by a comma, the respective certainty levels for each element in the lists of (6.1) and (6.2), coming from (4) and (5) respectively, also as a list of strings (ie. ""50"", ""95"", ""85)
6.4. Add a column called ""other names"", where you create another list of strings, where the information of each super natural being is separated by a semi-colon, where you add, if any, other names that a deity identified receives, or write ""missing"" if there are no other names of that super natural being according to the paragraph. If a deity has multiple other names, separate them with a comma (ie. missing; ""God of the sky"", ""He who is all powerful""; missing)
6.5. For each of the deity categories in (2) except 2.20 create a column with a list of strings, where each element of the string represents a deity, in the same order as (6.1). Within each column the list of strings should include 1s and 0s. The number one should be used when the deity in that position qualifies as being part of the category of that column. If the deity does not qualify in that category, the element corresponding to it should be the number zero. Column names should be: cat_creator_universe, cat_creator_human, cat_mother, cat_wife, cat_primal, cat_omni, cat_present, cat_absent, cat_warrior, cat_nature, cat_cosmos, cat_death, cat_ruler, cat_dual, cat_trick, cat_evil, cat_good, cat_demigod, cat_inter, cat_other_specify.
6.6. For the category ""other-specify"", also use a list of strings but assigning the additional deity category of 2.20 instead of ones and zeros. If there is not an additional category, write 0 for each deity.

7. Add an additional column called cat_type that classifies each supernatural being using a list of strings as before, into ""individual"" if there is only one deity by that name, ""multiple"" if the type of deity identified corresponds to multiple beigns, and ""missing"" if you can't tell from the information from the paragraph. 

8. Create a column called inception_myth where you write 1 if the paragraph is part of a myth of creation, or conception of the world or humankind, and 0 otherwise. "

At the end, return only a valid JSON object containing the following fields (with lists for each deity if applicable):
```json
{{
  "deities": "ExampleDeity",
  "gender": female,
  "certainty_deity": 95,
  "certainty_gender": 90,
  "other names": missing,
  "cat_creator_universe": 1,
  "cat_creator_human": 0,
  "cat_mother": 0,
  "cat_wife": 0,
  "cat_primal": 0,
  "cat_omni": 0,
  "cat_present": 1,
  "cat_absent": 0,
  "cat_warrior": 0,
  "cat_nature": 0,
  "cat_cosmos": 1,
  "cat_death": 0,
  "cat_ruler": 0,
  "cat_dual": 0,
  "cat_trick": 0,
  "cat_evil": 0,
  "cat_good": 1,
  "cat_demigod": 0,
  "cat_inter": 0,
  "cat_other_specify": 0,
  "cat_type": individual,
  "inception_myth": 1
}}

"Respond with only the JSON object. Do not include explanations, prefaces, or markdown formatting. Begin your response with a curly brace and end with a closing curly brace."
."""

        try:
            client = openai.OpenAI(api_key=openai.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
            )

            reply = response.choices[0].message.content
            print(f"\n--- Raw GPT Response ---\n{reply}\n------------------------\n")
            row_data = extract_structured_data(reply)
            row_data["uuid"] = row["uuid"]
            result_df = pd.concat([result_df, row_data], ignore_index=True)

        except Exception as e:
            print(f"Error on row {idx}: {e}")
            continue

    # Save progress
    result_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
