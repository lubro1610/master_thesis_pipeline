# The thought behind this script is to take the PDF MPR and the web-based and consolidate into a single CSV file with a standardized format.
# This is useful because the pipeline then just needs to read from one file for the entire MPR category. It also makes adjustments easier,
# as we could just append new documents to the consolidated file if we change the scope of the analysis for some reason.
# This stage is pure data manipulation, which is illustrated by the the imports as well.

import pandas as pd
import os

# This docstring defines the standardized format I chose for all the documents in the consolidated file. It is a simple format,
# but it is great for the purpose of the thesis, as it cleanly separates the necessary metadata. 
# The first iteration had "date", which I later changed to "exact_date", and the prior is just an artifcant of development
# Date is not kept for the output file, only exact_date.
def consolidate_mpr_to_standard_format():
    """
    Consolidates cleaned legacy PDF MPR data and web MPR data into standard CSV format:
    "date","url","title","category","exact_date","full_text_raw"
    """
    
    # Notice how this step joins legacy_pdf_clean and web_mpr. As you would have seen in 03_clean, there was no need for a separate cleaning
    # script in the web-based MPRs, as we could just target the wanted HTML elements and disregards the others.
    # Read cleaned legacy PDF data
    legacy_path = os.path.join("02_Data", "raw_text", "raw_text_mpr_legacy_pdf_CLEAN.csv")
    df_legacy = pd.read_csv(legacy_path)
    
    # Read web MPR data
    web_path = os.path.join("02_Data", "raw_text", "raw_text_web_mpr.csv")
    df_web = pd.read_csv(web_path)
    
    print(f"legacy_pdf_documents={len(df_legacy)}")
    print(f"web_documents={len(df_web)}")
    
    df_legacy_std = pd.DataFrame()
    
    # Map columns: Full_Text_Clean (instead of Full_Text_Raw)
    df_legacy_std["date"] = "Year: " + df_legacy["date"].str.extract(r'(\d{4})', expand=False)
    df_legacy_std["url"] = df_legacy["URL"]
    df_legacy_std["title"] = df_legacy["Title"]
    df_legacy_std["category"] = df_legacy["Source"]
    df_legacy_std["exact_date"] = None  # Will fill below
    df_legacy_std["full_text_raw"] = df_legacy["Full_Text_Clean"]  # Use cleaned text
    
    print(f"legacy_pdf_shape={df_legacy_std.shape}")
    
    df_web_std = pd.DataFrame()
    
    # Extract year from Date column
    df_web_std["date"] = "Year: " + df_web["Date"].str.extract(r'(\d{4})', expand=False)
    df_web_std["url"] = df_web["URL"]
    df_web_std["title"] = df_web["Title"]
    df_web_std["category"] = "WEB_HTML"  # Mark as web source
    df_web_std["exact_date"] = None  # Will fill below
    df_web_std["full_text_raw"] = df_web["Full_Text_Raw"]
    
    print(f"web_shape={df_web_std.shape}")
    
    # CONSOLIDATION 
    # You may notice that I use pd.concat here, which is a simple way to stack the two dataframes, even though it doesnt sort them in any way.
    # its not really necessary anyway, but I will probably sort at a later stage so that I can manually check the corpus without losing complete track.
    consolidated = pd.concat([df_legacy_std, df_web_std], ignore_index=True)
    
    # Check for overlaps (same URL or same title+date). This is an important step to ensure we dont get any duplicates, which is a risk
    # when doing this sort of two-step consolidation. Luckily, this is quite an easy process at this stage, as were not working with too
    # many documents, and we know that Q3 2021 is the final PDF report before the web-regime takes over.
    print(f"before_deduplication={len(consolidated)}")
    
    # Remove duplicates based on URL
    consolidated = consolidated.drop_duplicates(subset=["url"], keep="first")
    
    print(f"after_deduplication={len(consolidated)}")
    
    # OUTPUT 
    # I know it may look kinda confusing how im going back and forth between raw_text and clean, and to be fair I get confused myself sometimes.
    # Probably not the best naming convention here, I must admit
    output_path = os.path.join("02_Data", "clean_consolidated", "raw_text_mpr_consolidated.csv")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save with proper column order and quoting
    consolidated = consolidated[["date", "url", "title", "category", "exact_date", "full_text_raw"]]
    consolidated.to_csv(output_path, index=False, encoding='utf-8-sig', quoting=1)  # quoting=1 is QUOTE_ALL
    
    print(f"\nsaved: {output_path}")
    print(f"total_documents={len(consolidated)}")
    
    # Summary
    print(f"exact_date_present={consolidated['exact_date'].notna().sum()}")
    print(f"exact_date_missing={consolidated['exact_date'].isna().sum()}")

if __name__ == "__main__":
    consolidate_mpr_to_standard_format()
