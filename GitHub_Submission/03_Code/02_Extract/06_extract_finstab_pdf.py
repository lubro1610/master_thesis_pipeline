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

# PDF-specific filters
# Tougher filters for PDFs since tables often "bleed" into text
PDF_NOISE = [
    r"^\d+\s*Financial Stability.*?\d+",          # Page headers
    r"Norges Bank\s*\|.*?\|",                      # Footers
    r"Appendix\s+\d+.*",                           # Appendices
    r"Source:\s+.*",                               # Source citations
    r"^\d+$",                                      # Pure page numbers
    r"^\s*-{3,}\s*$",                              # Separator lines
    r"Financial stability\s+\d+/\d+",              # Report identifiers
    r"ISSN\s+\d{4}-\d{4}",                         # ISSN numbers
]

def is_table_dense(text, threshold=0.35):
    """Checks if a text block has too many numbers (table detection)."""
    clean = text.strip()
    if not clean:
        return True
    digits = sum(c.isdigit() for c in clean)
    return (digits / len(clean)) > threshold

def extract_pdf_blocks(pdf_content):
    """Uses coordinate-based block reading to handle columns correctly."""
    full_text = []
    
    try:
        with fitz.open(stream=io.BytesIO(pdf_content), filetype="pdf") as doc:
            # Skip first 2-3 pages (Cover, ToC) and last 2 (Appendices)
            # For very old PDFs (< 10 pages) skip less
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
                
                # Sort blocks: First by y0 (top to bottom), then by x0 (left to right)
                # This ensures we follow column flow
                blocks.sort(key=lambda b: (b[1], b[0]))

                for b in blocks:
                    block_text = b[4].strip()
                    
                    # Quality control on block
                    if len(block_text) < 15:
                        continue  # Ignore small fragments
                    if is_table_dense(block_text):
                        continue  # Ignore number-heavy rows
                    
                    # Run PDF-specific noise cleaning
                    for pattern in PDF_NOISE:
                        block_text = re.sub(pattern, "", block_text, flags=re.IGNORECASE | re.MULTILINE)
                    
                    if block_text.strip():
                        full_text.append(block_text.strip())

        return "\n\n".join(full_text)
    except Exception as e:
        return f"PDF_ERROR: {e}"

def extract_metadata(soup):
    """Extract date from landing page metadata."""
    metadata = {"date": "Unknown"}
    
    # Look for article:published_time meta tag
    date_tag = soup.find('meta', attrs={'property': 'article:published_time'})
    if date_tag:
        date_content = date_tag.get('content', '')
        # Parse date format: "10/29/2018 10:00:00 AM" -> "2018-10-29"
        date_part = date_content.split()[0] if ' ' in date_content else date_content
        try:
            from datetime import datetime
            dt = datetime.strptime(date_part, '%m/%d/%Y')
            metadata["date"] = dt.strftime('%Y-%m-%d')
        except:
            metadata["date"] = date_part  # Fallback to original
    
    return metadata

def find_pdf_link(soup, landing_url):
    """
    Finds PDF link using multiple patterns for different years.
    Handles 2018, 2010, and 2000-style button structures.
    """
    pdf_url = None
    
    # Pattern 1: 2018 style - Look for <dl class="micro-defs"> followed by PDF link
    # The PDF link is usually nearby in a download section
    micro_defs = soup.find('dl', class_='micro-defs')
    if micro_defs:
        # Look for "Financial Stability Report" series
        series_dd = micro_defs.find('dd', string=re.compile(r'Financial Stability', re.I))
        if series_dd:
            # Find PDF link in nearby sections
            pdf_link = soup.find('a', href=re.compile(r'\.pdf', re.IGNORECASE))
            if pdf_link:
                pdf_url = pdf_link.get('href')
    
    # Pattern 2: 2010/2000 style - <p class="download-link download-link--enhanced">
    if not pdf_url:
        download_link = soup.find('p', class_='download-link')
        if download_link:
            pdf_link = download_link.find('a', href=re.compile(r'\.pdf', re.IGNORECASE))
            if pdf_link:
                pdf_url = pdf_link.get('href')
    
    # Pattern 3: Generic fallback - any link with "financial" and ".pdf"
    if not pdf_url:
        pdf_link = soup.find('a', href=re.compile(r'financial.*?\.pdf', re.IGNORECASE))
        if pdf_link:
            pdf_url = pdf_link.get('href')
    
    # Pattern 4: Last resort - any PDF link
    if not pdf_url:
        pdf_link = soup.find('a', href=re.compile(r'\.pdf', re.IGNORECASE))
        if pdf_link:
            pdf_url = pdf_link.get('href')
    
    return pdf_url

def process_landing_page(landing_url, headers):
    """Finds and fetches PDF from landing page, extracts metadata and text."""
    try:
        res = requests.get(landing_url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.content, 'html.parser')
        
        # Extract metadata (date) from landing page
        metadata = extract_metadata(soup)
        
        # Find PDF link using multiple patterns
        pdf_url = find_pdf_link(soup, landing_url)

        if pdf_url:
            # Ensure absolute URL
            pdf_url = urljoin("https://www.norges-bank.no", pdf_url)
            
            # Download PDF
            pdf_res = requests.get(pdf_url, timeout=20)
            text = extract_pdf_blocks(pdf_res.content)
            
            return {
                'text': text,
                'date': metadata['date']
            }
        else:
            return {
                'text': "NO_PDF_LINK_FOUND",
                'date': "Unknown"
            }
    
    except Exception as e:
        return {
            'text': f"LANDING_PAGE_ERROR: {e}",
            'date': "Unknown"
        }

def extract_year_from_date(date_str):
    """Extracts year from date string (format: YYYY-MM-DD or similar)."""
    match = re.search(r'(\d{4})', date_str)
    if match:
        return int(match.group(1))
    return None

def run_financial_stability_pdf_scrape(csv_path, mode='all'):
    """
    Scrapes legacy Financial Stability reports (PDF) from 2018 and earlier.
    
    Args:
        csv_path: Path to urls_pubs_financial_stability.csv
        mode: 'all' = all PDF reports (2018 and earlier)
    """
    df_urls = pd.read_csv(csv_path)
    
    # Filter: 2018 and earlier (PDF-only reports)
    # Extract year from 'date' column
    df_urls['year'] = df_urls['date'].apply(extract_year_from_date)
    df_legacy = df_urls[df_urls['year'] <= 2018].copy()
    
    print(f"total_reports={len(df_urls)}")
    print(f"pdf_reports={len(df_legacy)}")
    print(f"web_reports={len(df_urls) - len(df_legacy)}")
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    results = []
    successful = 0
    failed = 0

    for idx, row in tqdm(df_legacy.iterrows(), total=len(df_legacy), desc="Extracting"):
        result = process_landing_page(row['url'], headers)
        
        # Extract text and date from result
        text = result['text']
        exact_date = result['date']
        
        # Check if extraction was successful
        if text and not text.startswith("NO_PDF_LINK") and not text.startswith("LANDING_PAGE_ERROR") and not text.startswith("PDF_ERROR"):
            successful += 1
        else:
            failed += 1
        
        results.append({
            "url": row['url'],
            "title": row['title'],
            "category": "Financial Stability",
            "exact_date": exact_date,
            "full_text_raw": text
        })
        
        # Be gentle with the server - PDFs are heavy files
        time.sleep(1.5)

    if results:
        df_out = pd.DataFrame(results)
        output_path = os.path.join("02_Data", "raw_text", "raw_text_financial_stability_pdf.csv")
        
        df_out.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig', lineterminator='\n')
        
        print(f"\nsaved: {output_path}")
        print(f"total_reports={len(df_out)}")
        print(f"successful={successful}")
        print(f"failed={failed}")
        
        # Text statistics for successful extractions
        successful_texts = [r['full_text_raw'] for r in results 
                           if not r['full_text_raw'].startswith(("NO_PDF_LINK", "LANDING_PAGE_ERROR", "PDF_ERROR"))]
        if successful_texts:
            lengths = [len(t) for t in successful_texts]
            print(f"text_length_min={min(lengths):,}")
            print(f"text_length_median={int(pd.Series(lengths).median()):,}")
            print(f"text_length_max={max(lengths):,}")
if __name__ == "__main__":
    csv_path = os.path.join("02_Data", "urls", "urls_pubs_financial_stability.csv")
    run_financial_stability_pdf_scrape(csv_path, mode='all')
