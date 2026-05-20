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
    # Remove source lines (Sources:) - both standalone lines and with trailing text
    r"(?:^|\n)\s*Sources:\s+.*?(?=\n|$)",
    # Remove percent-lines that often trail tables
    r"(?:^|\n)\s*Percent(?:\s|$)",
    r"(?:^|\n)\s*Percentage\s+change(?:\s|$)",
]

def extract_metadata(soup):
    metadata = {"date": "Unknown", "title": "Unknown"}
    date_tag = soup.find('meta', attrs={'property': 'og:published_time'})
    if date_tag:
        metadata["date"] = date_tag.get('content', '').split('T')[0]
    title_tag = soup.find('title')
    if title_tag:
        metadata["title"] = title_tag.get_text().strip()
    return metadata

def extract_clean_web_report(soup):
    """Removes tables and figure descriptions from the HTML."""
    
    # 1. DECOMPOSE: Remove unwanted elements entirely from the soup
    # This removes figure captions (figcaption), all tables (table), and figure elements with images
    for noise_tag in soup.find_all(['figure', 'figcaption', 'table', 'nav', 'header', 'footer', 'script', 'style']):
        noise_tag.decompose()
    
    # Remove only footnote content (tooltips), not the buttons themselves
    # Remove divs with class containing "tool-tip__explained" or "tool-tip__content"
    for tooltip_content in soup.find_all('div', class_=re.compile(r'tool-tip__explained|tool-tip__content', re.I)):
        tooltip_content.decompose()
    
    # Remove specific table containers used by Norges Bank (but keep publication-infobox with text)
    for table_div in soup.find_all('div', class_=re.compile(r'publication-chapter__table', re.I)):
        table_div.decompose()

    full_content = []
    
    # 2. FIND CHAPTERS
    # So compared to press releases and speeches, MPR web reports have a more complex structure.
    # However, they are still far more joyful to work with than the PDFs, as we can target specific elements.
    chapters = soup.find_all('section', class_='publication-chapter__chapter')
    if not chapters:
        main_content = soup.select_one('main#main-content') or soup.body
        chapters = [main_content]

    for chapter in chapters:
        # Get headings, paragraphs and infoboxes
        # We use a selector that ignores everything inside 'cover' images
        elements = chapter.find_all(['h1', 'h2', 'p', 'div'], recursive=True)
        
        for el in elements:
            # Sjekk om elementet er en infoboks eller vanlig tekst
            is_infobox = 'publication-infobox' in el.get('class', [])
            is_intro = 'intro' in el.get('class', [])
            
            if el.name in ['h1', 'h2', 'p'] or is_infobox or is_intro:
                # separator=" " sikrer at ord ikke klistres sammen hvis det er spans/links inni
                txt = el.get_text(separator=" ").strip()
                
                # Avoid short fragments and duplicates
                if len(txt) > 20 and txt not in full_content:
                    full_content.append(txt)

    # 3. ASSEMBLY AND CLEANING
    text = "\n\n".join(full_content)
    text = text.replace('"', "'") # Prevent CSV format breakage
    
    # Use MULTILINE and IGNORECASE flags for better regex matching (NOT DOTALL - it makes . too greedy)
    for pattern in NOISE_PHRASES:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove excessive line breaks created by the cleaning
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove lines containing only whitespace after regex cleaning
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if line.strip()]
    text = '\n'.join(cleaned_lines)
    
    # Rekonstruer paragraf-separatorer (minst 2 linjer mellomrom)
    text = re.sub(r'\n\n+', '\n\n', text)
    
    return text.strip()

def run_validation_to_csv(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    output_dir = os.path.join("02_Data", "raw_text")
    if not os.path.exists(output_dir): os.makedirs(output_dir)
        
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.content, 'html.parser')
        
        # 1. Finn Web-Report knappen (håndterer både ny og gammel struktur)
        web_btn = (
            soup.select_one('a.visual-publication__link--primary') or 
            soup.find('a', class_='btn-one', string=re.compile(r'web edition', re.I)) or
            soup.find('a', string=re.compile(r'Read the report', re.I))
        )
        if web_btn:
            report_url = urljoin("https://www.norges-bank.no", web_btn['href'])
            
            report_res = requests.get(report_url, headers=headers, timeout=10)
            report_soup = BeautifulSoup(report_res.content, 'html.parser')
            
            # Run metadata extraction and text extraction
            meta = extract_metadata(soup)
            final_text = extract_clean_web_report(report_soup)
            
            return {
                "URL": url,
                "Title": meta["title"],
                "Date": meta["date"],
                "Full_Text_Raw": final_text
            }
        else:
            print("no_web_report_button")
            return None

    except Exception as e:
        print(f"error={e}")
        return None

def scrape_all_mpr_reports(start_year=2021, end_year=2025):
    """Scrapes all web-based MPR reports from start_year to end_year."""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    output_dir = os.path.join("02_Data", "raw_text")
    if not os.path.exists(output_dir): 
        os.makedirs(output_dir)
    
    # Generate all MPR URLs for the period (try both URL patterns)
    mpr_urls = []
    for year in range(end_year, start_year - 1, -1):  # From newest to oldest
        for quarter in range(4, 0, -1):  # 4, 3, 2, 1
            # Try both URL patterns: mpr-42021 and mpr-4-2021
            url1 = f"https://www.norges-bank.no/en/news-events/publications/Monetary-Policy-Report/{year}/mpr-{quarter}{year}/"
            url2 = f"https://www.norges-bank.no/en/news-events/publications/Monetary-Policy-Report/{year}/mpr-{quarter}-{year}/"
            mpr_urls.append((url1, url2))  # Lagrer begge varianter
    
    print(f"mpr_urls={len(mpr_urls)}, period={start_year}-{end_year}")
    
    all_data = []
    successful = 0
    failed = 0
    
    for i, (url1, url2) in enumerate(mpr_urls, 1):
        print(f"[{i}/{len(mpr_urls)}]", end=" ")
        
        # Try url1 first, then url2 if url1 does not work
        result = run_validation_to_csv(url1)
        if not result:
            result = run_validation_to_csv(url2)
        
        if result:
            all_data.append(result)
            successful += 1
        else:
            failed += 1
        
        print()  # Ny linje etter hver rapport
        time.sleep(1)  # Vent 1 sekund mellom hver rapport (vennlig scraping)
    
    # Lagre alle resultater i én CSV
    if all_data:
        df = pd.DataFrame(all_data)
        output_path = os.path.join(output_dir, "raw_text_web_mpr.csv")
        df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, encoding='utf-8-sig', lineterminator='\n')
        
        print(f"\nsaved: {output_path}")
        print(f"successful={successful}/{len(mpr_urls)}")
        print(f"failed={failed}")
    else:
        print("\nno_reports_fetched")

if __name__ == "__main__":
    # Scrape all web-based reports from 2021-2025
    scrape_all_mpr_reports(start_year=2021, end_year=2025)