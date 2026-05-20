# Package structure is pretty much identical throughout all the extraction scripts, as the logic is inherently similar.
# Please look to 01_extract_speeches.py for details on the overall package structure and the main extraction logic.
# You may find artifacts across the extraction scripts (such as logic for speaker extraction etc, even though only speeches have speakers)
# This was a slight shortcut to streamline the process, but it does not affect the final output in any way, as the logic is only applied if relevant. 
# 04_extract_mpr_pdf.py contains some notes on the extraction logic of the older (after Q3 2021) PDF reports.

import pandas as pd
import requests
from bs4 import BeautifulSoup
import os
import time
import io
import re
import csv
import fitz  # PyMuPDF
from tqdm import tqdm
from urllib.parse import urljoin

def extract_metadata(soup, url):
    """
    Extracts date and speaker from <head> meta tags.
    Falls back to regex when tags are missing.
    """
    metadata = {
        "exact_date": "Unknown Date",
        "speaker": "Norges Bank"
    }
    
    # 1. DATE EXTRACTION (Meta Tags)
    date_tag = soup.find('meta', attrs={'property': 'article:published_time'}) or \
               soup.find('meta', attrs={'property': 'og:published_time'})
    
    if date_tag and date_tag.get('content'):
        metadata["exact_date"] = date_tag['content'].split(' ')[0]
    else:
        date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})', url)
        if date_match:
            metadata["exact_date"] = f"{date_match.group(2)}/{date_match.group(3)}/{date_match.group(1)}"

    # 2. SPEAKER EXTRACTION
    # Check Meta Tags (Common in 2012, 1998)
    author_tag = soup.find('meta', attrs={'property': 'article:author'}) or \
                 soup.find('meta', attrs={'name': 'author'})
    
    if author_tag and author_tag.get('content'):
        metadata["speaker"] = author_tag['content'].strip()
    
    # Check Profile Name (Modern Speeches)
    if metadata["speaker"] == "Norges Bank":
        profile_tag = soup.find(class_='profile-image-block__name')
        if profile_tag:
            metadata["speaker"] = profile_tag.get_text().strip()

    # THE "BACHE" FIX: Regex search in Intro/Description
    # We look for the Governor's name in the full descriptive string
    if metadata["speaker"] == "Norges Bank":
        intro = soup.find(class_='intro')
        desc = soup.find('meta', attrs={'property': 'og:description'})
        search_text = (intro.get_text() if intro else "") + (desc.get('content', '') if desc else "")
        
        # Greedy regex to capture multi-part names (e.g., Ida Wolden Bache)
        # Captures capitalized words until a common separator like 'at', 'on', or 'in'.
        name_match = re.search(r'(?:Governor|statement by|Address by)\s+([A-Z][a-zæøå\-\.]+(?:\s+[A-Z][a-zæøå\-\.]+){1,3})', search_text)
        if name_match:
            metadata["speaker"] = name_match.group(1).strip()

    return metadata

def extract_clean_web_content(soup):
    """Extracts the article body and removes noise elements."""
    content_parts = []
    article = soup.select_one('article.article') or soup.select_one('#article') or soup.select_one('main')
    if not article: return ""

    title = article.find('h1')
    if title: content_parts.append(title.get_text().strip())

    intro = article.find(class_='intro')
    if intro: content_parts.append(intro.get_text().strip())

    body = article.find(class_=re.compile("article__main-body|c-article__body", re.I))
    if body:
        for noise in body.find_all(class_=re.compile("no-print|visually-hidden|share|tags|feedback", re.I)):
            noise.decompose()
        
        text = body.get_text(separator="\n\n").strip()
        
        # Scrub residual document noise that dilutes sentiment analysis
        noise_phrases = [
            r"Download presentation.*?\(pdf\)",
            r"Chart\s+\d+[:\s].*?\n",
            r"Check against delivery",
            r"Please note that the text below may differ from the actual address",
            r"Show image.*?\n",
            r"Read more about.*?on behalf of the government\.?",
            r"Share.*?\n",                                    
            r"Follow us.*?\n",                                 
            r"Related.*?:|See also.*?:",                       
            r"Print.*?page|Email.*?article",                   
            r"©\s*\d{4}.*?\n",                                 
            r"Last updated.*?:\s*.*?\n",                       
            r"Table of contents.*?\n",                        
            r"Feedback|Tell us what you think",                
            r"Filed under.*?:|Category:.*?:|Tags:.*?\n",                                     
            r"^\s*-{3,}\s*$",                                 
            r"Breadcrumb.*?\n|Home\s*>\s*.*?>",              
            r"\bEnglish\b|\bNorsk\b",                   
            r"Archive.*?:|Previous.*?:|Next.*?:",            
            r"Contact.*?:|Phone:.*?\n",                        
            r"Subscribe.*?to.*?newsletter",                    
            r"^\s*\[\s*(?:Photo|Image|Chart|Table)\s*\].*?\n",
            r"For more information.*?(?:Q&A|FAQs?|links?|resources?|website)\.?",
            r"For (?:more|further) information.*?(?:Q&A|FAQs?|Norges Bank|links?|resources?|website)\.?",
            r"Show chart.*?\n"
        ]
        for pattern in noise_phrases:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
            
        content_parts.append(text.strip())

    return "\n\n".join([p for p in content_parts if p])

def process_url(landing_page_url, title_from_csv):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    base_url = "https://www.norges-bank.no"
    
    try:
        response = requests.get(landing_page_url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        meta = extract_metadata(soup, landing_page_url)
        
        web_btn = soup.select_one('a.visual-publication__link--primary')
        pdf_btn = soup.select_one('a.visual-publication__link--secondary')

        body_text = ""
        if web_btn:
            res = requests.get(urljoin(base_url, web_btn['href']), headers=headers)
            body_text = extract_clean_web_content(BeautifulSoup(res.content, 'html.parser'))
        elif pdf_btn:
            res = requests.get(urljoin(base_url, pdf_btn['href']))
            with fitz.open(stream=io.BytesIO(res.content), filetype="pdf") as doc:
                body_text = "\n".join([page.get_text() for page in doc])
        else:
            body_text = extract_clean_web_content(soup)

        # Sanitize Quotes to prevent CSV format breakage (The Quote-Collision Fix)
        safe_body = str(body_text).replace('"', "'")
        safe_title = str(title_from_csv).replace('"', "'")
        safe_speaker = str(meta['speaker']).replace('"', "'")

        # Construct a clean, audit-ready header inside the text cell
        organized_entry = (
            f"SOURCE: NORGES BANK PRESS RELEASE\n"
            f"TITLE: {safe_title}\n"
            f"SPEAKER: {safe_speaker}\n"
            f"DATE: {meta['exact_date']}\n"
            f"URL: {landing_page_url}\n"
            f"{'-'*40}\n\n"
            f"{safe_body}"
        )

        return {
            "full_text": organized_entry,
            "exact_date": meta["exact_date"],
            "speaker": safe_speaker
        }
    except Exception as e:
        return {"full_text": f"Error: {str(e)}", "exact_date": "Error", "speaker": "Error"}

def run_targeted_extraction(category_filename):
    input_dir = os.path.join("02_Data", "urls")
    output_dir = os.path.join("02_Data", "raw_text")
    input_path = os.path.join(input_dir, category_filename)
    if not os.path.exists(input_path): return

    df = pd.read_csv(input_path)
    print(f"input_rows={len(df)}")
    
    dates, speakers, texts = [], [], []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        result = process_url(row['url'], row['title'])
        dates.append(result["exact_date"])
        speakers.append(result["speaker"])
        texts.append(result["full_text"])
        time.sleep(0.7)

    df['exact_date'] = dates
    df['speaker'] = speakers
    df['full_text_raw'] = texts
    
    # Save with UTF-8-SIG for perfect Norwegian character rendering in Excel
    output_path = os.path.join(output_dir, category_filename.replace("urls_", "raw_text_"))
    
    # Use lineterminator to ensure every row is cleanly closed in the CSV file
    df.to_csv(
        output_path, 
        index=False, 
        quoting=csv.QUOTE_ALL, 
        encoding='utf-8-sig', 
        lineterminator='\n'
    )
    print(f"saved: {output_path}")

if __name__ == "__main__":
    run_targeted_extraction("urls_press_releases.csv")