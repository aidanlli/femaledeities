# Data Scraping For Female Deities

## Overview
- **Description**: Scraping text and sources from the eHRAF database.
- **Author**: Aidan Li
- **Version**: 1.1.0
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

Initialize a python environment and run 
```
pip install -r requirements.txt
```


# Data Scraping From eHRAF

## Overview

This project consists six Python scripts that collectively handle data scraping, processing, cleaning, output generation, and high-level metrics of every culture within the eHRAF World Cultures Database, which is maintained by the Human Relations Area Files (HRAF) at Yale University. While utilizing JavaScript may be faster, we used Python for the ability to display the scraping process in a manner that allows the user to interact with the scraping and visually verify bugs and missed cultures. The total runtime to scrape all information from the database is around 11 hours: cultural_sources_scraping.py, takes roughly three hours to run and paragraph_text_append.py takes approximately eight hours to run, but may vary depending on hardware. The other files have a comparatively trivial runtime. 

The file "qrySummary_eHRAF_WorldCultures_Jan2024" is manually downloaded from the HRAF website: "Cultures in eHRAF World Cultures", found [here](https://hraf.yale.edu/resources/reference/). However, I have also added two additional cultures that have been catalouged between the latest update of that file and 2/5/2025: Tarascans and Chiriguano. If you choose to download from the website directly instead of utilizing the file in the repository, make sure to update this information.
## Files

### 1. `cultural_sources_scraping.py` - **Obtaining paragraph source information from eHRAF**
- **Purpose**: To scrape all relevant paragraphs given culture and subject search strings
- **Key Functions**: search_ehraf: Initiates a program which automatically opens a Google Chrome tab, directs the search to the exact culture page based upon the subject filters, and selects and downloads all paragraph sources into a .csv file.
- **Dependencies**: pandas, selenium, time, urllib, os
- **Output**: 355 .csv files beginning with "ehrafSearch"

### 2. `cultural_sources_concatenate.py` - **Merging all individual dataframes together**
- **Purpose**: Takes every downloaded file from deityscraping.py and concatenates them into one master dataframe.
- **Key Functions**: count_and_concatenate: Define the expected columns in each df, count number of files, concatenate all dataframes, remove any possible duplicate entries based on paragraph UUID.
- **Dependencies**: os, pandas.

### 3. `cultural_sources_checks.py` - **Verifying that all culture files were correctly downloaded**
- **Purpose**: To verify the presence of all cultures, and to identify any missing cultures from the 
- **Key Functions**: Find cultures in the eHRAF list of cultures.
- **Dependencies**: pandas.

### 4. `paragraph_text_append.py` - **Taking our master dataframe and scraping the text information**
- **Purpose**: Scrape paragraph data for all UUIDs and append them to a new file.
- **Key Functions**: scrape_text: takes the link from column "Permalink" in the master dataframe and extracts paragraph information from each link.
- **Dependencies**: pandas, requests, bs4, tqdm, os

### 5. `verify_dataframe_accuracy.py` - **Verifying cohesion and accuracy of the newly scraped dataframe with the original source dataframe**
- **Purpose**: To verify that all uuids in the paragraph source dataframe are in the updated dataframe containing paragraph text

**How to Run**: 

1. Change the "csv_path" to wherever you downloaded the file qrySummary_eHRAF_WorldCultures_Jan2024.
2. Run the code using the following command. If you would like to visually see the process, please comment out "options.add_argument('--headless')" on line 26.
```
python deityscraping.py
```
3. Once the code finishes running, comment out options.add_argument('--headless'). Then, uncomment "#cultures = [ "O'odham","Mi'kmaq", "Chiricahua Apache"]" and comment out "cultures = df["EHRAF WORLD CULTURES NAME"] # Remove NaN values and duplicates". Then, for cultures O'odham and Mi'kmaq, you may have to help the macro navigate to the page displaying paragraphs. Alternatively, download these manually.
4. Run the concatenatedeities.py and duplicatecheck.py, only after you do so should you navigate back here (if applicable)
5. There are two cultures which the "download all" button does not work, even when pressed manually; this may be due to the quantity of these paragraphs. These are cultures Dogon and Ifugao; I've linked their pages [here](https://ehrafworldcultures.yale.edu/search/traditional/data?owcs=FA16&culture=Dogon&docs=28&sres=4365&q=cultures%3A%22Dogon%22+AND+subjects%3A%28%22spirits+and+gods%22+OR+%22gender+roles+and+issues%22+OR+%22mythology%22+OR+%22gender+status%22+OR+%22revelation+and+divination%22%29) and [here](https://ehrafworldcultures.yale.edu/search/traditional/data?owcs=OA19&culture=Ifugao&docs=22&sres=3610&q=cultures%3A%22Ifugao%22+AND+subjects%3A%28%22spirits+and+gods%22+OR+%22gender+roles+and+issues%22+OR+%22mythology%22+OR+%22gender+status%22+OR+%22revelation+and+divination%22%29), respectively. Instead, manually navigate to these pages and download these files in batches; I found that 2-3 seperate batches was sufficient.
6. If there are any other cultures missing after running duplicatecheck.py, you may either put these missing cultures into a list and set cultures = [your list here] and run the code again, or you can simply download the files directly from the website. After downloading these extra files, place them into the folder_path and run concatenatedeities.py and duplicatecheck.py. After this, you should get the expected output.



**How to Run**: 

1. If not already in a folder, move all donwloaded files to a folder. Do not move any files not produced by the previous deityscraping.py.
2. change the folder_path and the download_path to your liking
3. run the code using the following command:
```
python concatenatedeities.py
```


**How to Run**:

1. Change csv_path and concatenated_csv_path to your paths, as well as missing_cultures_path to a path of your choice.
2. Run the code using the following command:
```
python duplicatecheck.py
```
3. Examine the output of the code:
- If you see '['Eastern Apache', 'Baluchi', 'Karakalpak', 'Hazara', 'Ghorbat', 'Tajiks']', you are done. All the files have been correctly concatenated. You may proceed to the next file. Eastern Apache is now coded as Chiricahua Apache. Given the subject strings, all the other cultures do not have a file relevant to the search strings.
- If you see cultures OTHER than these, refer back to deityscraping step 5 and 6.


**How to Run**: 
1. Change input_path (result file from concatenatedeities.py) and output_path to your desired files and file paths
2. Run the code using the following command:
```
python deity_text.py
```
3. Come back in 8 hours (sleep, do other work, etc.)

Note: if you attempt to change the updating structure of the code (df.to_csv), when you run the code keep an eye on the size of the new file for several minutes. If you append incorrectly, the file size will massively increase to ~2gb in just several minutes.

