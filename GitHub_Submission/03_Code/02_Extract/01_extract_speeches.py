# For the extraction stage, pandas is used to read the URL-list of harvested publications, and to save/build the dataset.
# requests is used to fetch the content of the webpages, and is more practical than Selenium for this stage as we dont have to interact with anything.
# BeautifulSOup parses HTML content and lets us navigate/search the strucutre. 
# os is used for management of file paths and directories.
# time is used to add delays between requests, which is important in order not to be flagged as a bot and of course to be fair to Norges Bank's servers.
# io is used to open pdfs directly from memory without saving to disk, although this is not used for web-based speeches etc. Mainly for the strategic reports, but I just added them all.
# re is regex for textual matching and noise removal.
# csv is used to save the final dataset in a clean format.
# fitz (PyMuPDF) is the PDF-reader used in these scripts.
# tqdm is simply a progress bar to see the extraction process. I wanted to know how far Id come along in the process, but it is not essential.
# urljoin combines relative urls with base, such as when we find a link to a pdf in the harvested link, we combien with the base url to get the full url.

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

# The next few lines are helper functions for the main extraction function.
# There is a variety of different methods for extracting the relevant info, as the structure of the webpages varies as we move in time, 
# as well as differences between the publication types.
def extract_metadata(soup, url): 
    """
    Extracts date and speaker from <head> meta tags.
    Falls back to regex when tags are missing.
    """
    metadata = {
        "exact_date": "Unknown Date",
        "speaker": "Norges Bank"
    }
    
    # 1. DATE EXTRACTION (Meta Tags), main approach for finding the exact date, which is obviously crucial.
    date_tag = soup.find('meta', attrs={'property': 'article:published_time'}) or \
               soup.find('meta', attrs={'property': 'og:published_time'}) 
    
    if date_tag and date_tag.get('content'): 
        metadata["exact_date"] = date_tag['content'].split(' ')[0] 
    else:
        date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})', url) # Fallback to regex if meta tags are missing/takes some obscure format.
        if date_match:                                           # Obviously not that robust, but it can be helpful, and I absolutely check this manually for error entries.
            metadata["exact_date"] = f"{date_match.group(2)}/{date_match.group(3)}/{date_match.group(1)}" # American date format here, and Im not actually sure if this is consistent throuhgout scripts.
                                                                                                            # If it isnt, apologies to any potential reader...

    # 2. SPEAKER EXTRACTION
    # Check meta tags (based on observed differences throughout the years)
    author_tag = soup.find('meta', attrs={'property': 'article:author'}) or \
                 soup.find('meta', attrs={'name': 'author'})
    
    if author_tag and author_tag.get('content'):
        metadata["speaker"] = author_tag['content'].strip() # If we find a meta tag with author/name, we use that as the speaker. This is checked manually for correctness. 
    
    # Check Profile Name (Modern Speeches)
    if metadata["speaker"] == "Norges Bank":
        profile_tag = soup.find(class_='profile-image-block__name') # In more recent speeches there may be profile pictures attached, usually containing names.
        if profile_tag:                                             # As such, it represents a nice fallback.
            metadata["speaker"] = profile_tag.get_text().strip()

    # We look for the Governor's name in the full descriptive string
    if metadata["speaker"] == "Norges Bank":
        intro = soup.find(class_='intro')
        desc = soup.find('meta', attrs={'property': 'og:description'})
        search_text = (intro.get_text() if intro else "") + (desc.get('content', '') if desc else "")
        
        # Greedy regex to capture multi-part names (e.g., Ida Wolden Bache)
        # Captures capitalized words until a common separator like 'at', 'on', or 'in'. Last kind of desperate fallback
        name_match = re.search(r'(?:Governor|statement by|Address by)\s+([A-Z][a-zæøå\-\.]+(?:\s+[A-Z][a-zæøå\-\.]+){1,3})', search_text)
        if name_match:
            metadata["speaker"] = name_match.group(1).strip()

    return metadata

def extract_clean_web_content(soup):
    """Extracts the article body and removes noise elements."""
    content_parts = []
    article = soup.select_one('article.article') or soup.select_one('#article') or soup.select_one('main') # Look for main article body with common tags, but with fallbacks as structure can vary.
    if not article: return ""

    title = article.find('h1') # Title is often found in <h1> tag, and can be useful to include in corpus for better readability when doing manual checks. 
    if title: content_parts.append(title.get_text().strip()) 

    intro = article.find(class_='intro') 
    if intro: content_parts.append(intro.get_text().strip())

    body = article.find(class_=re.compile("article__main-body|c-article__body", re.I)) # Main body is typically found in a common class, but it varies and so the script uses regex.
    if body:
        for noise in body.find_all(class_=re.compile("no-print|visually-hidden|share|tags|feedback", re.I)): # Look for common noise elements and remove them, as they can dilute the analysis.
            noise.decompose()                                                                                # This is obviously easier in the filtering stage, but still wanted to remove as much as possible.
        
        text = body.get_text(separator="\n\n").strip() # Get the text with paragraphs separated by newlines, stripping whitespace. Just for manual readability when checking corpus. 
        
        # Scrub residual document noise that dilutes sentiment analysis. Tried to get rid of common elements with a bit of hardcoded regexes.
        noise_phrases = [
            r"Download presentation.*?\(pdf\)",
            r"Chart \d+.*?\n",
            r"Check against delivery",
            r"Please note that the text below may differ from the actual address",
            r"Show image.*?\n",
            r"Show chart.*?\n"
        ]
        for pattern in noise_phrases:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE) 
            
        content_parts.append(text.strip()) # Append the "cleaned" body text to content parts.

    return "\n\n".join([p for p in content_parts if p]) 

# Main function to process each URL, extract content, and organize in a clean format.
def process_url(landing_page_url, title_from_csv):   
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    base_url = "https://www.norges-bank.no"
    
    try:
        response = requests.get(landing_page_url, headers=headers, timeout=15) 
        soup = BeautifulSoup(response.content, 'html.parser') 
        meta = extract_metadata(soup, landing_page_url) 
        
        # Look for web version or PDF version of the speech. Web is easier.
        web_btn = soup.select_one('a.visual-publication__link--primary') 
        pdf_btn = soup.select_one('a.visual-publication__link--secondary') 

        body_text = ""
        if web_btn: # if we find web vesion, we prioritize that.
            res = requests.get(urljoin(base_url, web_btn['href']), headers=headers)
            body_text = extract_clean_web_content(BeautifulSoup(res.content, 'html.parser'))
        elif pdf_btn: # If we cant find one, we look for the other. 
            res = requests.get(urljoin(base_url, pdf_btn['href']))
            with fitz.open(stream=io.BytesIO(res.content), filetype="pdf") as doc: # if its a pdf, we extract the text with PyMuPDF.
                body_text = "\n".join([page.get_text() for page in doc]) 
        else:
            body_text = extract_clean_web_content(soup) # Backup if neither option is available (edit: this backup is never used)

        # Clean quotes to prevent CSV format breakage, and replace " with ' specifically.
        safe_body = str(body_text).replace('"', "'")
        safe_title = str(title_from_csv).replace('"', "'")
        safe_speaker = str(meta['speaker']).replace('"', "'")

        # Construct a clean header inside the text cell
        organized_entry = (
            f"SOURCE: NORGES BANK SPEECH ARCHIVE\n"
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
    except Exception as e: # In case of error, return error instead of crashing script. Allows for manual revision of any individual error while allowing the process to continue.
        return {"full_text": f"Error: {str(e)}", "exact_date": "Error", "speaker": "Error"}

# This function reads the harvested urls and processes each one, saving the final dataset with extracted text and metadata.
def run_targeted_extraction(category_filename):
    input_dir = os.path.join("02_Data", "urls")
    output_dir = os.path.join("02_Data", "raw_text")
    input_path = os.path.join(input_dir, category_filename)
    if not os.path.exists(input_path): return

    df = pd.read_csv(input_path)
    print(f"input_rows={len(df)}")
    
    # Loops through each URL with a progress bar just for practicality, and pauses 0.7 seconds between each request.
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
    run_targeted_extraction("urls_speeches.csv") # the urls_speeches.csv file is the output from the harvesting stage, contains urls of speeches.