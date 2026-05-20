# The consolidation scripts are really quite simple, as it is a simple steg in the pipeline. However, if you still would like to understand
# the logic behind it, I included some comments in 01_consolidate_mpr.py, which was the first implementation. 

import pandas as pd
import os

def consolidate_bank_lending():
    """Merge web and PDF Bank Lending Survey extractions into one dataset."""
    
    # File paths
    web_file = os.path.join("02_Data", "raw_text", "raw_text_bank_lending_survey.csv")
    pdf_file = os.path.join("02_Data", "raw_text", "raw_text_bank_lending_survey_pdf.csv")
    output_file = os.path.join("02_Data", "clean_consolidated", "raw_text_bank_lending_survey_consolidated.csv")
    
    # Read both files
    df_web = pd.read_csv(web_file, encoding='utf-8-sig')
    print(f"web_reports={len(df_web)}")
    
    df_pdf = pd.read_csv(pdf_file, encoding='utf-8-sig')
    print(f"pdf_reports={len(df_pdf)}")
    
    # Concatenate
    df_consolidated = pd.concat([df_web, df_pdf], ignore_index=True)
    print(f"before_deduplication={len(df_consolidated)}")
    
    # Remove duplicates based on URL (in case any overlap)
    df_consolidated = df_consolidated.drop_duplicates(subset=['url'], keep='first')
    print(f"after_deduplication={len(df_consolidated)}")
    
    # Filter out entries with "Unknown" dates
    unknown_count = (df_consolidated['exact_date'] == 'Unknown').sum()
    if unknown_count > 0:
        print(f"Removing {unknown_count} reports with unknown dates")
        df_consolidated = df_consolidated[df_consolidated['exact_date'] != 'Unknown']
    
    # Convert date to datetime for proper sorting
    df_consolidated['exact_date'] = pd.to_datetime(df_consolidated['exact_date'])
    
    # Sort by date (most recent first)
    df_consolidated = df_consolidated.sort_values('exact_date', ascending=False)
    
    # Convert date back to string format for CSV
    df_consolidated['exact_date'] = df_consolidated['exact_date'].dt.strftime('%Y-%m-%d')
    
    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_consolidated.to_csv(output_file, index=False, encoding='utf-8-sig', quoting=1)
    
    print(f"\nsaved: {output_file}")
    print(f"date_range={df_consolidated['exact_date'].min()} to {df_consolidated['exact_date'].max()}")
    print(f"total_reports={len(df_consolidated)}")
    
    # Show text length statistics
    df_consolidated['text_length'] = df_consolidated['full_text_raw'].str.len()
    print(f"text_length_min={df_consolidated['text_length'].min():,}")
    print(f"text_length_median={df_consolidated['text_length'].median():.0f}")
    print(f"text_length_max={df_consolidated['text_length'].max():,}")

if __name__ == "__main__":
    consolidate_bank_lending()
