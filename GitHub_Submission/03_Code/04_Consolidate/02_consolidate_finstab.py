# The consolidation scripts are really quite simple, as it is a simple steg in the pipeline. However, if you still would like to understand
# the logic behind it, I included some comments in 01_consolidate_mpr.py, which was the first implementation. 

import pandas as pd
import os

def consolidate_financial_stability_reports(use_cleaned_pdf=True):
    """
    Consolidate Financial Stability reports from web (2019-2025) and PDF (2000-2018).

    Args:
        use_cleaned_pdf: If True, use full_text_clean from PDF cleaning output.
                         If False, use full_text_raw from PDF output.
    """
    
    # Again, looking over this naming shcematic, I have to apologize to any potential reader. Its a slight mess, 
    # but the logic is quite simple. web_path is the output from web extraction, pdf_path is the outfrom the PDF cleaning stage,
    # and pdf_raw_path is the output from the PDF extraction stage. This is just a backup, and it is never utilized
    base_raw = os.path.join("02_Data", "raw_text")
    base_clean = os.path.join("02_Data", "Cleaned")

    web_path = os.path.join(base_raw, "raw_text_web_fin_stab.csv")
    pdf_path = os.path.join(base_raw, "raw_text_pdf_fin_stab_CLEAN.csv")
    pdf_raw_path = os.path.join(base_raw, "raw_text_pdf_fin_stab.csv")

    if not os.path.exists(web_path):
        raise FileNotFoundError(f"Missing web CSV: {web_path}")

    if use_cleaned_pdf:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Missing cleaned PDF CSV: {pdf_path}")
        df_pdf = pd.read_csv(pdf_path)
        pdf_text_col = "full_text_clean"
    else:
        if not os.path.exists(pdf_raw_path):
            raise FileNotFoundError(f"Missing raw PDF CSV: {pdf_raw_path}")
        df_pdf = pd.read_csv(pdf_raw_path)
        pdf_text_col = "full_text_raw"

    df_web = pd.read_csv(web_path)

    # Normalize PDF columns
    df_pdf_std = pd.DataFrame()
    df_pdf_std["url"] = df_pdf["url"]
    df_pdf_std["title"] = df_pdf["title"]
    df_pdf_std["category"] = df_pdf.get("category", "Financial Stability")
    df_pdf_std["exact_date"] = df_pdf["exact_date"]
    df_pdf_std["full_text_raw"] = df_pdf[pdf_text_col]

    # Normalize Web columns
    df_web_std = pd.DataFrame()
    df_web_std["url"] = df_web["url"]
    df_web_std["title"] = df_web["title"]
    df_web_std["category"] = df_web.get("category", "Financial Stability")
    df_web_std["exact_date"] = df_web["exact_date"]
    df_web_std["full_text_raw"] = df_web["full_text_raw"]

    # Consolidate
    consolidated = pd.concat([df_web_std, df_pdf_std], ignore_index=True)

    # Drop duplicates on URL if any
    consolidated = consolidated.drop_duplicates(subset=["url"], keep="first")

    # Sort by date (newest first)
    consolidated["exact_date"] = pd.to_datetime(consolidated["exact_date"], errors="coerce")
    consolidated = consolidated.sort_values(by="exact_date", ascending=False)
    consolidated["exact_date"] = consolidated["exact_date"].dt.strftime("%Y-%m-%d")

    # Ensure output directory exists
    os.makedirs(base_clean, exist_ok=True)

    output_path = os.path.join(base_clean, "raw_text_financial_stability_consolidated.csv")
    consolidated.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"saved: {output_path}")
    print(f"rows={len(consolidated)}")

if __name__ == "__main__":
    consolidate_financial_stability_reports(use_cleaned_pdf=True)
