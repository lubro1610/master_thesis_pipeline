# Package structure is pretty much identical throughout all the extraction scripts, as the logic is inherently similar.
# Please look to 01_extract_speeches.py for details on the overall package structure and the main extraction logic.
# You may find artifacts across the extraction scripts (such as logic for speaker extraction etc, even though only speeches have speakers)
# This was a slight shortcut to streamline the process, but it does not affect the final output in any way, as the logic is only applied if relevant. 
# 04_extract_mpr_pdf.py contains some notes on the extraction logic of the older (after Q3 2021) PDF reports.

import fitz  # PyMuPDF
import re
import pandas as pd
import os
import csv
import io
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from urllib.parse import urljoin

# PDFs are typically even noisier than the web-based reports, and I tried to remove as much as possible even at the extraction stage.
# Although it proved much more efficient at later stages, I imagined I would be able to achieve more with these filters
# than what turned out to be the case. They are still somewhat useful, though...
PDF_NOISE = [
    r"^\d+\s*Monetary Policy Report\s+\d+/\d+",  # Page headers
    r"Norges Bank\s*\|.*?\|",                     # Page footers
    r"Appendix\s+\d+.*",                          # Appendices
    r"Source:\s+.*",                              # Source references in PDF
    r"^\d+$",                                     # Bare page numbers
    r"Inflation Report",                          # Old name for MPR
    r"^\s*-{3,}\s*$"                              # Separator lines
]

# A heuruistic function to detect if a text block is likely a table. For PDFs, I couldnt simply disregard certain web elements, which made a clean extraction...
# slightly more difficult. I experimented a bit with finding a good threhold for the ratio of digits to total characters, and found 0.35 to be a good balance.
# Obviously PDF filtering is never perfect, and some tabular data is likely to slip through, while some text may be accidentally removed.
# However, I believe this approach is still an improvement over leaving all tables in. I believe the 35% threshold to still be conservative.
def is_table_dense(text, threshold=0.35):
    """Checks if a text block has too many digits (table detection)."""
    clean = text.strip()
    if not clean:
        return True
    digits = sum(c.isdigit() for c in clean)
    return (digits / len(clean)) > threshold

# This is really the core of the PDF extraction logic. I initially used get_text(), as it simply reads text line by line. However, it blends lines together, and really creates a mess.
# PDFs such as the MPR has a two-column layout, and get_text() doesnt respect this, and doesnt produce the wanted result. Using blocks instead allows us to read the PDF in a more structured way.
def extract_pdf_blocks(pdf_content):
    """Uses coordinate-based block reading to handle columns correctly."""
    full_text = []
    
    try:
        with fitz.open(stream=io.BytesIO(pdf_content), filetype="pdf") as doc:
            # Skip the first 2-3 pages (cover, ToC) and last 2 (appendices)
            # For very old PDFs (< 10 pages) skip fewer to avoid losing content
            if len(doc) > 10:
                start_page = 2
                end_page = len(doc) - 2
            else:
                start_page = 0
                end_page = len(doc)

            for page_num in range(start_page, end_page):
                page = doc[page_num]
                # "blocks" returns (x0, y0, x1, y1, "text", block_no, block_type)
                blocks = page.get_text("blocks")
                
                # Sort blocks: first by y0 (top to bottom), then by x0 (left to right)
                # This ensures we follow the column flow, instead of reading across columns which get_text() would.
                blocks.sort(key=lambda b: (b[1], b[0]))

                for b in blocks:
                    block_text = b[4].strip()
                    
                    # Quality control on the block
                    if len(block_text) < 15:
                        continue  # Ignore small fragments, typically like single-worded elements and page numbers.
                    if is_table_dense(block_text):
                        continue  # Ignore table-like rows 
                    
                    # Run PDF-specific noise cleaning (aka, all the hard coded regex patterns I found to be common)
                    # Obviously not perfect or exhaustive, but it should reduce noise a bit.
                    for pattern in PDF_NOISE:
                        block_text = re.sub(pattern, "", block_text, flags=re.IGNORECASE | re.MULTILINE)
                    
                    if block_text.strip(): # Only add non-empty blocks after attempted cleaning.
                        full_text.append(block_text.strip())

        return "\n\n".join(full_text) 
    except Exception as e:
        return f"PDF_ERROR: {e}"

def process_landing_page(landing_url, headers): # Finds PDF link on landing page of any given report, runs PDF extraction. 
    """Finds and fetches the PDF from the landing page."""
    try:
        res = requests.get(landing_url, headers=headers, timeout=15) 
        soup = BeautifulSoup(res.content, 'html.parser') 
        
        # Find the PDF link (usually the secondary link on MPR pages)
        pdf_btn = (
            soup.select_one('a.visual-publication__link--secondary') or 
            soup.find('a', href=re.compile(r'\.pdf', re.IGNORECASE)) # Fallback to any link containing .pdf if class selector fails.
        )

        if pdf_btn and pdf_btn.get('href'):
            pdf_url = urljoin("https://www.norges-bank.no", pdf_btn['href'])
            pdf_res = requests.get(pdf_url, timeout=20)
            return extract_pdf_blocks(pdf_res.content)
        else:
            return "NO_PDF_LINK_FOUND"
    
    except Exception as e:
        return f"LANDING_PAGE_ERROR: {e}"

# At a certain point for all strategic reports, they are no longer available in a web format, but only as PDFs. 
# As such, this logic was important to ensure that we located the cutoff, which I found to be 2021 Q3. 
# Reports prior to this are only available as PDFs, while reports from 2021 Q4 and newer are available in a web format.
def extract_year_quarter(url):
    """Extracts year and quarter from URL."""
    # Try multiple patterns for year
    year_match = re.search(r'/(\d{4})/', url)
    if year_match:
        year = int(year_match.group(1))
        # Get quarter - can be "1", "2", "3", "4", "1/12", etc.
        quarter_match = re.search(r'[/\-]([1-4])(?:/\d{2})?(?:/)?$', url)
        quarter = int(quarter_match.group(1)) if quarter_match else 0
        return year, quarter
    return None, None

# Test function to check that PDF-extraction produces the wanted results for different time periods.
def find_representative_samples(df_urls): 
    """Finds representative samples from 2021, 2016, 2012, 2005, 1998."""
    target_years = [2021, 2016, 2012, 2005, 1998]
    samples = []
    
    for target_year in target_years:
        # Find at least one document from each year
        for idx, row in df_urls.iterrows():
            year, quarter = extract_year_quarter(row['url'])
            if year == target_year:
                samples.append(row)
                print(f"sample_year={target_year}: {row['title']}")
                break
    
    return samples

def run_legacy_mpr_scrape(csv_path, mode='sample'):
    """
    Scrapes legacy MPR reports (PDF) from 1999 to 2021 Q2.
    
    Args:
        csv_path: Path to urls_pubs_monetary_policy_report.csv
        mode: 'sample' = representative years (4 PDFs), 'all' = all legacy PDFs (2021 Q3 to 1998)
    """
    df_urls = pd.read_csv(csv_path)
    
    # Filter: From 2021 Q3 and back to 1998
    # 2021 Q3 URL contains "mpr-32021"
    # I exclude 2021 Q4 ("mpr-42021") and newer (these have HTML versions)
    df_legacy = df_urls.copy()
    
    # Chooses reports from 2021 Q3 and older. Everything more recent has web versions. 
    def is_legacy_pdf(url):
        year, quarter = extract_year_quarter(url)
        if year is None:
            return False
        # 2021 Q3 and older (back to 1998)
        if year < 2021:
            return True
        if year == 2021 and quarter <= 3:  # Q1, Q2, Q3
            return True
        return False
    
    # Filter to only legacy PDFs based on URL patterns.
    df_legacy = df_legacy[df_legacy['url'].apply(is_legacy_pdf)].copy()
    
    # sample for testing 
    if mode == 'sample':
        samples_rows = find_representative_samples(df_legacy)
        df_to_scrape = pd.DataFrame(samples_rows)
        output_suffix = "_SAMPLE"
    else: # the default full mode, extraction of all "legacy" PDFs
        df_to_scrape = df_legacy
        output_suffix = ""
    print(f"pdf_reports={len(df_to_scrape)}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    results = []

    for idx, row in tqdm(df_to_scrape.iterrows(), total=len(df_to_scrape)): 
        text = process_landing_page(row['url'], headers) 
        
        results.append({
            "date": row['date'],
            "URL": row['url'],
            "Title": row['title'],
            "Source": "PDF_LEGACY",
            "Full_Text_Raw": text
        })
        
        # Be polite to the server 
        time.sleep(1.5)

    # Save results to CSV with proper quoting to handle any special characters.
    if results:
        df_out = pd.DataFrame(results)
        output_path = os.path.join("02_Data", "raw_text", f"raw_text_mpr_legacy_pdf{output_suffix}.csv")
        
        df_out.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig')
        print(f"\nsaved: {output_path}")
        print(f"rows={len(df_out)}")

# runs full extraction of all legacy PDFs (MPR)
if __name__ == "__main__":
    csv_path = os.path.join("02_Data", "urls", "urls_pubs_monetary_policy_report.csv")
    
    # MODE: 'sample' = 4 representative years, 'all' = all legacy PDFs (2021 Q3 to 1998)
    run_legacy_mpr_scrape(csv_path, mode='all')
