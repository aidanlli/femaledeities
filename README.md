# Data Scraping For Female Deities

## Overview
- **Description**: Scraping text and sources from the eHRAF database.
- **Author**: Aidan Li
- **Date**: 2/20/2025
- **Version**: 1.2.0
- **Repository**: [https://github.com/aidanlli/femaledeities](https://github.com/aidanlli/femaledeities)

## Setup & Installation

### Prerequisites
  - "requests"
  - "bs4"
  - "pandas"
  - "tqdm"
  - "selenium"
  - "urllib"
  - "webdriver_manager"
  - "os"
  - "time"
  - "matplotlib"

Initialize a python environment and run 
```
pip install -r requirements.txt
```


# Data Scraping From eHRAF

## Overview

This project consists six Python scripts that collectively handle data scraping, processing, cleaning, output generation, and high-level metric generation of every culture within the eHRAF World Cultures Database, which is maintained by the Human Relations Area Files (HRAF) at Yale University. While utilizing JavaScript may be faster, we used Python for the ability to display the scraping process in a manner that allows the user to interact with the scraping and visually verify bugs and missed cultures. The total runtime to scrape all information from the database is around 11 hours: cultural_sources_scraping.py, takes roughly three hours to run and paragraph_text_append.py takes approximately eight hours to run, but may vary depending on hardware. The other files have a comparatively trivial runtime. 

The file "qrySummary_eHRAF_WorldCultures_Jan2024" is manually downloaded from the HRAF website: "Cultures in eHRAF World Cultures", found [here](https://hraf.yale.edu/resources/reference/). However, I have also added two additional cultures that have been catalouged between the latest update of that file and 2/5/2025: Tarascans and Chiriguano. If you choose to download from the website directly instead of utilizing the file in the repository, make sure to update this information.
## Files

### 1. `cultural_sources_scraping.py` - **Obtaining paragraph source information from eHRAF**
- **Purpose**: To scrape all relevant paragraphs given culture and subject search strings
- **Key Functions**: search_ehraf: Initiates a program which automatically opens a Google Chrome tab, directs the search to the exact culture page based upon the subject filters (), and selects and downloads all paragraph sources into a .csv file.
- **Dependencies**: pandas, selenium, time, urllib, os
- **Output**: 355 .csv files beginning with "ehrafSearch"
- **Subject Filters**: cult of the dead, general character of religion, cosmology, mythology, animism, eschatology, spirits and gods, sacred objects and places, theological systems, revelation and divination, luck and chances

### 2. `cultural_sources_concatenate.py` - **Merging all individual dataframes together**
- **Purpose**: Takes every downloaded file from deityscraping.py and concatenates them into one master dataframe.
- **Key Functions**: count_and_concatenate: Define the expected columns in each df, count number of files, concatenate all dataframes, remove any possible duplicate entries based on paragraph UUID.
- **Dependencies**: os, pandas.
- **Output**: One large .csv file (~66mb) containing all files from `cultural_sources_scraping.py` output, as well as manually incorporated .csvs.

### 3. `cultural_sources_checks.py` - **Verifying that all culture files were correctly downloaded**
- **Purpose**: To verify the presence of all cultures, and to identify any missing cultures from the 
- **Key Functions**: Find cultures in the eHRAF list of cultures.
- **Dependencies**: pandas.
- **Output**: A list of cultures present in the culture master list but not present in the large .csv file produced from `cultural_sources_concatenate.py`.
### 4. `paragraph_text_append.py` - **Taking our master dataframe and scraping the text information**
- **Purpose**: Scrape paragraph data for all UUIDs and append them to a new file.
- **Key Functions**: scrape_text: takes the link from column "Permalink" in the master dataframe and extracts paragraph information from each link.
- **Dependencies**: pandas, requests, bs4, tqdm, os
- **Output**: An appended version of the large .csv file produced in `cultural_sources_scraping.py` with two extra rows: "Raw Text" and "Text". The new .csv is around 470mb.

### 5. `verify_dataframe_accuracy.py` - **Verifying cohesion and accuracy of the newly scraped dataframe with the original source dataframe**
- **Purpose**: To verify that all uuids in the paragraph source dataframe are in the updated dataframe containing paragraph text.
- **Key Functions**: Check if all uuids in .csv produced by `paragraph_text_append.py` are in `cultural_sources_concatenate.py`, remove rows in that don't have a matching uuid. Counts blank rows in Raw Text and Text, counts number of rows in Raw Text and Text with "No text found".
- **Dependencies**: pandas, re
- **Output**: 8 print statements describing dataframe row removal and possible missing data in "Raw Text" and "Text".

### 6. `dataframe_high_level_metrics` - **Creating plots and tables summarizing dataframe**
- **Purpose**: To summarize high-level metrics and interesting preliminary statistics of the dataframe
- **Key Functions**: Create 8 tables and 8 plots showing distribution of key aspects of the dataframe, such as Region, Culture, Substinence, and Keyword distribution.
- **Dependencies**: pandas, re, matplotlib, numpy, seaborn, math
- **Output**: 8 tables and 8 plots - 7 bar charts, 1 histogram.

### 7. `text_cleaning_subject_expansion` - **cleaning text and extracting subject tags for each source**
- **Purpose**: To prepare for analysis by cleaning the "text" column and identifying subject tags in each source
- **Key Functions**: Remove extraneous text from "Text" column, primarily the phrase "Load in Context". Create 11 new columns that show whether or not a source has a certain tag, and creates one bar chart and one table displaying the distribution.
- **Dependencies**: pandas, matplotlib
- **Output**: Cleaned "Text" column, 11 new columns, 1 table, 1 bar chart

## **How to Run**: 
1. Change the "csv_path" in `cultural_sources_scraping` to wherever you downloaded the file qrySummary_eHRAF_WorldCultures_Jan2024. 
2. Run the following command. If you would like to visually see the process, please comment out "options.add_argument('--headless')" on line 26. in `cultural_sources_scraping.py`.
```
python cultural_sources_scraping.py
```
3. Move all downloaded files, as well as the files under the "files" folder on Github to a folder of your choice. Open `cultural_sources_concatenate.py` and change the download_path and folder_path to your desired paths. The folder_path should correspond to the folder your placed all the downloaded files into, while the download_path is up to you.
4. Run the following code
```
python cultural_sources_concatenate.py
```
5. Change the "csv_path" in `cultural_sources_checks.py` to the path from step 1. Change `concatenated_csv_path` to wherever the output of `cultural_sources_concatenate.py` is located.
6. Run the following code
```
python cultural_sources_checks.py
```
If you see the following output:

Missing culture: Dominicans

Missing culture: Eastern Apache

Missing culture: Turkmens

Missing culture: Hazara

Missing culture: Pamir Peoples

Ignore steps 7-8 and immediately move to step 9. If you see additional missing cultures, create a list with the additional missed cultures. For example, if we are missing the Yoruba, Iroquois, and Ainu, the list should be as follows:
['Yoruba', 'Iroquois', 'Ainu']

7. Take the list created and navigate back to `cultural_sources_scraping`. Uncomment line 125 and paste your list, replacing the dummy list present. 
8. Run the following code:
```
python cultural_sources_scraping.py

python cultural_sources_concatenate.py

python cultural_sources_checks.py
```
You should now see the following output:

Missing culture: Dominicans

Missing culture: Eastern Apache

Missing culture: Turkmens

Missing culture: Hazara

Missing culture: Pamir Peoples

If not, repeat steps 7-8 with the updated list until the output matches above.

9. Open `paragraph_text_append` and change the "input_path" to the output of step 2 and the "output_path" to the output from step 6. Then, run the following code:
```
python paragraph_text_append.py
```
This should take about 8 hours.

10. Open `verify_dataframe_accuracy` and change the "input_path" and "output_path" to the paths from step 9. Then, run the following code:
```
python verify_dataframe_accuracy.py
```
Make sure that the column 'Raw Text' has no blank rows. If you do not see "All uuids from concatenated are present in updated" and instead see "Missing uuids in updated output:", this is a serious issue. If you cannot diagnose this problem, send me a message. Make sure that the number of rows with "No text found" is low; as of 2/20/2025, there should be 13 instances of "No text found" . If the number of rows is significantly higher than this, there may have been an issue with the paragraph appending from step 9. To ensure this is an error in the code, open the .csv output and verify whether or not the "No text found" rows truly have no text found when you view the Permalink. 

11. Open `dataframe_high_level_metrics` and change the "output_path" to the path from steps 9 and 10. Change the "export_path" to an export path of your choice that can store 16 .png files. Then, run the following code:
```
python dataframe_high_level_metrics.py
```
Check the graphs and tables to make sure they seem reasonable. The tables for Region, Subregion, Subsistence, Culture, and DocType should have a row titled "Total ___". These should all have the same value. The tables for male, female, and dual keywords also have a "Total __", but these will not be equivalent. The keywords are listed below:

Male Keywords:

he, him, his, father, king, lord, prince, son, husband, brother, patriarch, fatherhood, brotherhood

Female Keywords:

she, her, mother, queen, lady, goddess, daughter, wife, sister, matriarch, motherhood

Dual Keywords:

androgynous, dual-gendered, god/goddess

12. Open `text_cleaning_subject_expansion.py`and change the file paths as desired. Then, run the following code:
```
python text_cleaning_subject_expansion.py
```

## Final Output
Your final output should consist of the following:

1 ~66mb .csv file with columns "uuid", "Primary_Author", "Title", "Published", "Page", "DocType", "Culture", "Region", "Subregion", "Subsistence", "OCM", "IDs", "Permalink".

1 ~470mb .csv file with columns "uuid", "Primary_Author", "Title", "Published", "Page", "DocType", "Culture", "Region", "Subregion", "Subsistence", "OCM", "IDs", "Permalink", "Raw Text", "Text"

8 tables: Culture, DocType, dual_keywords, female_keywords, male_keywords, Region, Subregion, Subject type, and Subsistence
- These tables should list the exact counts of each type of Culture, DocType, Region, Subregion, Subject type, and Subsistence method, as well as the frequency of male keywords, female keywords, and dual keywords in "Text".

8 bar charts: DocType, dual_keywords, female_keywords, male_keywords, Region, Subregion, Subject type, and Subsistence
- These charts visually display the relative counts and frequency of each type of DocType, Region, Subregion, Subject type, and Subsistence and Male/Female/Dual keywords.

1 histogram: Culture 
- This histogram displays the distribution of the quantity of source paragraphs for each source.
