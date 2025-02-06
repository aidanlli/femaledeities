---
title: "Data Scraping For Female Deities"
description: "Scraping text and sources from the eHRAF database."
author: "Aidan Li"
version: "1.0.0"
repository: "https://github.com/aidanlli/femaledeities"
dependencies:
  - "requests"
  - "beautifulsoup4"
  - "pandas"
  - "tqdm"
  - "selenium"
  - "urllib"
  - "webdriver_manager"
  - "requests"
toc: true
---
# Data Scraping From eHRAF

## Overview

This project consists of four Python scripts that collectively handle data scraping, processing, and output generation of every culture within the eHRAF
World Cultures Database, which is maintained by the Human Relations Area Files (HRAF) at Yale University. While utilizing JavaScript may be faster, this method displays the process in a manner that allows the user to interact with the scraping and visually verify bugs and missed cultures. The total runtime to scrape all information from the database, including paragraph text, is around 11 hours: deityscraping.py, duplicatecheck.py, and concatenatedeities.py take rougly 3 hours to run, and deity_text.py takes approximately eight hours to run, but may vary depending on hardware.

The file "qrySummary_eHRAF_WorldCultures_Jan2024" is manually downloaded from the HRAF website: "Cultures in eHRAF World Cultures", found [here](https://hraf.yale.edu/resources/reference/). However, I have also added two additional cultures that have been catalouged between the latest update of that file and 2/5/2025: Tarascans and Chiriguano. If you choose to download from the website directly instead of utilizing the file in the repository, make sure to update this information.
## Files

### 1. `deityscraping.py` - **Obtaining paragraph information from eHRAF**
- **Purpose**: To scrape all relevant paragraphs given culture and subject search strings
- **Key Functions**: search_ehraf: Initiates a program which automatically opens a Google Chrome tab, directs the search to the exact culture page based upon the subject filters, and selects and downloads all paragraph sources into a .csv file.
- **Dependencies**: pandas, selenium, time, urllib, os

**How to Run**: 

1. Change the "csv_path" to wherever you downloaded the file qrySummary_eHRAF_WorldCultures_Jan2024.
2. Run the code. If you would like to visually see the process, please comment out "options.add_argument('--headless')".
3. Once the code finishes running, comment out options.add_argument('--headless'). Then, uncomment "#cultures = [ "O'odham","Mi'kmaq", "Chiricahua Apache"]" and comment out "cultures = df["EHRAF WORLD CULTURES NAME"] # Remove NaN values and duplicates". Then, for cultures O'odham and Mi'kmaq, you may have to help the macro navigate to the page displaying paragraphs. Alternatively, download these manually.
4. Run the concatenatedeities.py and duplicatecheck.py, only after you do so should you navigate back here (if applicable)
5. There are two cultures which the "download all" button does not work, even when pressed manually; this may be due to the quantity of these paragraphs. These are cultures Dogon and Ifugao. Instead, manually navigate to these pages and download these files in batches; I found that 2-3 seperate batches was sufficient.
6. If there are any other cultures missing after running duplicatecheck.py, you may either put these missing cultures into a list and set cultures = [your list here] and run the code again, or you can simply download the files directly from the website. After downloading these extra files, place them into the folder_path and run concatenatedeities.py and duplicatecheck.py. After this, you should get the expected output.


### 2. `concatenatedeities.py` - **Merging all individual dataframes together**
- **Purpose**: Takes every downloaded file from deityscraping.py and concatenates them into one master dataframe.
- **Key Functions**: count_and_concatenate: Define the expected columns in each df, count number of files, concatenate all dataframes, remove any duplicate entries.
- **Dependencies**: os, pandas.

**How to Run**: 

1. If not already in a folder, move all donwloaded files to a folder. Do not move any files not produced by the previous deityscraping.py.
2. change the folder_path and the download_path to your liking
3. run the code

### 3. `duplicatecheck.py` - **Verifying that all file were correctly downloaded**
- **Purpose**: To verify the presence of all cultures, and to identify any missing cultures from the 
- **Key Functions**: Find cultures in the eHRAF list of cultures.
- **Dependencies**: pandas.

**How to Run**:

1. Change csv_path and concatenated_csv_path to your paths, as well as missing_cultures_path to a path of your choice.
2. Run the code.
3. Examine the output of the code:
- If you see '['Eastern Apache', 'Baluchi', 'Karakalpak', 'Hazara', 'Ghorbat', 'Tajiks']', you are done. All the files have been correctly concatenated. You may proceed to the next file. Eastern Apache is now coded as Chiricahua Apache. Given the subject strings, all the other cultures do not have a file relevant to the search strings.
- If you see cultures OTHER than these, refer back to deityscraping step 5 and 6.

### 4. `deity_text.py` - **Taking our master dataframe and scraping the text information**
- **Purpose**: Scrape paragraph data for all UUIDs and append them to a new file.
- **Key Functions**: scrape_text: takes the link from column "Permalink" in the master dataframe and extracts paragraph information from each link.
- **Dependencies**: pandas, requests, bs4, tqdm, os

**How to Run**: 
1. Change input_path and output_path to your desired files and file paths
2. Run the code
3. Come back in 8 hours (sleep, do other work, etc.)

Note: if you attempt to change the updating structure of the code (df.to_csv), when you run the code keep an eye on the size of the new file for several minutes. If you append incorrectly, the file size will massively increase to ~2gb in just several minutes.

## Setup & Installation

### Prerequisites
  - "requests"
  - "beautifulsoup4"
  - "pandas"
  - "tqdm"
  - "selenium"
  - "urllib"
  - "webdriver_manager"
  - "os"
  - "time"
