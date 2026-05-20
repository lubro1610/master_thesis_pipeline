# Package structure is pretty much identical throughout all the extraction scripts, as the logic is inherently similar.
# Please look to 01_extract_speeches.py for details on the overall package structure and the main extraction logic.
# You may find artifacts across the extraction scripts (such as logic for speaker extraction etc, even though only speeches have speakers)
# This was a slight shortcut to streamline the process, but it does not affect the final output in any way, as the logic is only applied if relevant. 
# 04_extract_mpr_pdf.py contains some notes on the extraction logic of the older (after Q3 2021) PDF reports.

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import os
import csv
import time
from urllib.parse import urljoin
from tqdm import tqdm

# Noise filter for text (regex)
NOISE_PHRASES = [
    r"Download presentation.*?\(pdf\)",
    r"Chart\s+\d+[:\s].*?\n",
    r"Check against delivery",
    r"Please note.*?differ from the actual address",
    r"Show image.*?\n",
    r"Read more about.*?on behalf of the government\.?",
    r"Share.*?\n", r"Follow us.*?\n", r"Related.*?:|See also.*?:",
    r"Print.*?page|Email.*?article", r"©\s*\d{4}.*?\n",
    r"Last updated.*?:\s*.*?\n", r"Table of contents.*?\n",
    r"Feedback|Tell us what you think", r"^\s*-{3,}\s*$",
    r"Breadcrumb.*?\n|Home\s*>\s*.*?>", r"\bEnglish\b|\bNorsk\b",
    r"Archive.*?:|Previous.*?:|Next.*?:", r"Contact.*?:|Phone:.*?\n",
    r"^\s*\[\s*(?:Photo|Image|Chart|Table)\s*\].*?\n",
    r"Show chart.*?\n",
    # Remove table references (Table 1, Table 2, Table 2a etc.) - with flexible whitespace
    r"(?:^|\n)\s*Table\s+\d+[a-z]?\s+.*?(?=\n|$)",
    # Remove source lines (Sources:) - both on separate line and with text after
    r"(?:^|\n)\s*Sources:\s+.*?(?=\n|$)",
    # Remove percent-sign lines that often close tables
    r"(?:^|\n)\s*Percent(?:\s|$)",
    r"(?:^|\n)\s*Percentage\s+change(?:\s|$)",
]

def extract_metadata(soup):
    """Extract date and title from landing page metadata."""
    metadata = {"date": "Unknown", "title": "Unknown"}
    
    # Extract date from article:published_time meta tag
    date_tag = soup.find('meta', attrs={'property': 'article:published_time'})
    if date_tag and date_tag.get('content'):
        # Format: "11/5/2019 10:00:00 AM" -> extract date portion
        date_str = date_tag.get('content', '').strip()
        # Try to parse and convert to ISO format
        try:
            # Split on space and take first part (date only)
            date_part = date_str.split(' ')[0]
            # Convert from M/D/YYYY to YYYY-MM-DD
            from datetime import datetime
            dt = datetime.strptime(date_part, '%m/%d/%Y')
            metadata["date"] = dt.strftime('%Y-%m-%d')
        except:
            metadata["date"] = date_part  # Fallback to original
    
    # Extract title
    title_tag = soup.find('meta', attrs={'property': 'og:title'}) or soup.find('title')
    if title_tag:
        metadata["title"] = title_tag.get('content') if title_tag.name == 'meta' else title_tag.get_text()
        metadata["title"] = metadata["title"].strip()
    
    return metadata

def extract_clean_web_report(soup):
    """Extracts the article body and removes tables and figure descriptions."""
    
    # 1. DECOMPOSE: Delete unwanted elements completely from the "soup"
    for noise_tag in soup.find_all(['figure', 'figcaption', 'table', 'nav', 'header', 'footer', 'script', 'style']):
        noise_tag.decompose()
    
    # Remove ALL tooltip elements (both trigger buttons and popup content)
    for tooltip in soup.find_all('span', class_='tool-tip'):
        tooltip.decompose()
    
    # Also remove any remaining footnote backlink spans
    for footnote_span in soup.find_all('span', id=re.compile(r'footnote-\d+-backlink')):
        footnote_span.decompose()
    
    # Remove specific table containers that Norges Bank uses
    for table_div in soup.find_all('div', class_=re.compile(r'publication-chapter__table', re.I)):
        table_div.decompose()

    full_content = []
    
    # 2. FIND CHAPTERS
    chapters = soup.find_all('section', class_='publication-chapter__chapter')
    if not chapters:
        main_content = soup.select_one('main#main-content') or soup.body
        chapters = [main_content] if main_content else []

    for chapter in chapters:
        # First, extract infoboxes as complete units to avoid double-extraction
        infoboxes = chapter.find_all('div', class_='publication-infobox')
        for infobox in infoboxes:
            txt = infobox.get_text(separator=" ").strip()
            if len(txt) > 20 and txt not in full_content:
                full_content.append(txt)
        
        # Track which elements are inside infoboxes to skip them later
        infobox_elements = set()
        for infobox in infoboxes:
            for el in infobox.find_all(['h1', 'h2', 'h3', 'p']):
                infobox_elements.add(id(el))
        
        # Now extract all other paragraphs and headings that aren't in infoboxes
        for el in chapter.find_all(['h1', 'h2', 'h3', 'p'], recursive=True):
            if id(el) in infobox_elements:
                continue  # Skip elements already extracted as part of infoboxes
            txt = el.get_text(separator=" ").strip()
            if len(txt) > 20 and txt not in full_content:
                full_content.append(txt)

    # 3. ASSEMBLY AND WASH
    text = "\n\n".join(full_content)
    text = text.replace('"', "'")  # Prevent CSV crash
    
    # Use MULTILINE and IGNORECASE flags for better regex matching (NOT DOTALL - makes . too greedy)
    for pattern in NOISE_PHRASES:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove excess line breaks created by washing
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove lines that only contain whitespace after regex wash
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if line.strip()]
    text = '\n'.join(cleaned_lines)
    
    # Reconstruct paragraph separators (at least 2 line spaces)
    text = re.sub(r'\n\n+', '\n\n', text)
    
    return text.strip()

def process_financial_stability_url(landing_page_url, title_from_csv):
    """Process a single Financial Stability report URL."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    base_url = "https://www.norges-bank.no"
    
    try:
        # Fetch the landing page
        response = requests.get(landing_page_url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract metadata from landing page
        meta = extract_metadata(soup)
        
        # Find the "Read the report (web edition)" button
        # Looking for: <a href="..." class="btn btn-one">Read the report (web edition)</a>
        web_btn = soup.find('a', class_='btn-one', string=re.compile(r'Read the report.*web edition', re.I))
        
        # Also try Norwegian version
        if not web_btn:
            web_btn = soup.find('a', class_='btn-one', string=re.compile(r'Les rapporten.*nettformat', re.I))
        
        # Also try the visual-publication pattern used in MPR
        if not web_btn:
            web_btn = soup.select_one('a.visual-publication__link--primary')
        
        # Fallback: Some reports (e.g., 2021) just say "Read the report" without "web edition"
        if not web_btn:
            web_btn = soup.find('a', class_='btn-one', string=re.compile(r'Read the report', re.I))
        
        body_text = ""
        if web_btn and web_btn.get('href'):
            report_url = urljoin(base_url, web_btn['href'])
            
            # Fetch the actual report content
            report_res = requests.get(report_url, headers=headers, timeout=15)
            report_soup = BeautifulSoup(report_res.content, 'html.parser')
            
            # Extract clean text
            body_text = extract_clean_web_report(report_soup)
        else:
            # Fallback: try to extract from landing page directly
            body_text = extract_clean_web_report(soup)
        
        # Sanitize for CSV
        safe_body = str(body_text).replace('"', "'")
        safe_title = str(title_from_csv).replace('"', "'")
        
        return {
            "url": landing_page_url,
            "title": safe_title,
            "category": "Financial Stability",
            "exact_date": meta["date"],
            "full_text_raw": safe_body
        }
        
    except Exception as e:
        return {
            "url": landing_page_url,
            "title": title_from_csv,
            "category": "Financial Stability",
            "exact_date": "Error",
            "full_text_raw": f"Error: {str(e)}"
        }

def scrape_financial_stability_reports():
    """Scrape web-based Financial Stability reports (2019-2025 only)."""
    output_dir = os.path.join("02_Data", "raw_text")
    input_path = os.path.join("02_Data", "urls", "urls_pubs_financial_stability.csv")
    
    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    
    # Filter to only web-based reports (2019-2025)
    # Extract year from the 'date' column (format: "Year: YYYY")
    df['year_int'] = df['date'].str.extract(r'(\d{4})').astype(int)
    df_web = df[df['year_int'] >= 2019].copy()
    
    print(f"total_reports={len(df)}")
    print(f"web_reports={len(df_web)}")
    print(f"pdf_reports={len(df) - len(df_web)}")
    
    all_data = []
    successful = 0
    failed = 0
    
    for idx, row in tqdm(df_web.iterrows(), total=len(df_web), desc="Extracting"):
        result = process_financial_stability_url(row['url'], row['title'])
        all_data.append(result)
        
        if result['exact_date'] != "Error":
            successful += 1
        else:
            failed += 1
            print(f"\nWARNING: Failed: {row['title']}")
        
        time.sleep(1)  # Be polite to the server
    
    # Save results
    if all_data:
        output_df = pd.DataFrame(all_data)
        output_path = os.path.join(output_dir, "raw_text_financial_stability.csv")
        
        output_df.to_csv(
            output_path, 
            index=False, 
            quoting=csv.QUOTE_ALL, 
            encoding='utf-8-sig', 
            lineterminator='\n'
        )
        
        print(f"\nsaved: {output_path}")
        print(f"total_reports={len(all_data)}")
        print(f"successful={successful}")
        print(f"failed={failed}")
        
        # Show date coverage
        valid_dates = output_df[output_df['exact_date'] != 'Error']['exact_date']
        if len(valid_dates) > 0:
            print(f"date_range={valid_dates.min()} to {valid_dates.max()}")
        
        # Show text statistics
        lengths = output_df[output_df['exact_date'] != 'Error']['full_text_raw'].str.len()
        print(f"text_length_min={lengths.min():,}")
        print(f"text_length_median={int(lengths.median()):,}")
        print(f"text_length_max={lengths.max():,}")
    else:
        print("\nERROR: No reports were extracted!")

if __name__ == "__main__":
    scrape_financial_stability_reports()
