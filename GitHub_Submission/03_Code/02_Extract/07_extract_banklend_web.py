# Package structure is pretty much identical throughout all the extraction scripts, as the logic is inherently similar.
# Please look to 01_extract_speeches.py for details on the overall package structure and the main extraction logic.
# You may find artifacts across the extraction scripts (such as logic for speaker extraction etc, even though only speeches have speakers)
# This was a slight shortcut to streamline the process, but it does not affect the final output in any way, as the logic is only applied if relevant. 
# 04_extract_mpr_pdf.py contains some notes on the extraction logic of the older (after Q3 2021) PDF reports.

import csv
import os
import re
import time
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


NOISE_PATTERNS = [
    r"(?:^|\n)\s*Javascript is disabled\s*(?=\n|$)",
    r"(?:^|\n)\s*Due to this, parts of the content on this site will not be displayed\.?\s*(?=\n|$)",
    r"(?:^|\n)\s*Chart\s+\d+[A-Za-z]?\s*.*?(?=\n|$)",
    r"\(Chart\s+\d+[A-Za-z]?\)",  # Remove inline chart references like (Chart 1)
    r"(?:^|\n)\s*Downloads\s*(?=\n|$)",
    r"(?:^|\n)\s*Published\s+\d{1,2}\s+\w+\s+\d{4}.*?(?=\n|$)",
    r"(?:^|\n)\s*Print\s*(?=\n|$)",
    r"(?:^|\n)\s*Did you find what you were looking for\?.*?(?=\n|$)",
    r"(?:^|\n)\s*Give us feedback.*?(?=\n|$)",
    r"(?:^|\n)\s*Objective of the Bank Lending Survey\s*(?=\n|$)",
    r"(?:^|\n)\s*Charts\s*\(pdf\)\s*(?=\n|$)",
    r"(?:^|\n)\s*Data\s*\(xlsx\)\s*(?=\n|$)",
]


def extract_metadata(soup):
    """Extract exact date and title from landing page metadata."""
    metadata = {"date": "Unknown", "title": "Unknown"}

    date_tag = soup.find("meta", attrs={"property": "article:published_time"})
    if date_tag and date_tag.get("content"):
        date_str = date_tag.get("content", "").strip()
        try:
            date_part = date_str.split(" ")[0]
            dt = datetime.strptime(date_part, "%m/%d/%Y")
            metadata["date"] = dt.strftime("%Y-%m-%d")
        except Exception:
            metadata["date"] = date_str

    title_tag = (
        soup.find("meta", attrs={"property": "og:title"})
        or soup.find("h1")
        or soup.find("title")
    )
    if title_tag:
        metadata["title"] = (
            title_tag.get("content")
            if title_tag.name == "meta"
            else title_tag.get_text(strip=True)
        )

    return metadata


def extract_clean_lending_text(soup):
    """Extract and clean the web-based Bank Lending Survey report text."""
    for noise_tag in soup.find_all(
        [
            "figure",
            "figcaption",
            "table",
            "nav",
            "header",
            "footer",
            "script",
            "style",
            "form",
            "aside",
            "noscript",
        ]
    ):
        noise_tag.decompose()

    # Remove disclaimer sections
    for disclaimer in soup.find_all(["div", "section"], class_=re.compile(r"disclaimer", re.I)):
        disclaimer.decompose()

    # Remove downloads section entirely
    for section in soup.find_all(["section", "div"], class_=re.compile(r"download", re.I)):
        section.decompose()

    # Remove entire paragraphs containing footnote references (removes both link and text)
    # Collect parents first to avoid modifying while iterating
    parents_to_remove = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href", "")
        if href and href.lower().startswith("#_ftn"):
            parent = a_tag.find_parent(["p", "div", "li"])
            if parent and parent not in parents_to_remove:
                parents_to_remove.append(parent)
    
    for parent in parents_to_remove:
        parent.decompose()
    
    # Remove specific links (objective page + download files)
    links_to_remove = []
    for a_tag in soup.find_all("a", href=True):
        href = a_tag.get("href", "")
        if href:
            href_lower = href.lower()
            if (
                "objective-of-the-bank-lending-survey" in href_lower
                or href_lower.endswith(".pdf")
                or href_lower.endswith(".xlsx")
            ):
                links_to_remove.append(a_tag)
    
    for link in links_to_remove:
        link.decompose()

    main_content = soup.select_one("main#main-content") or soup.body
    if not main_content:
        return ""

    chunks = []
    for el in main_content.find_all(["h1", "h2", "h3", "p", "li"], recursive=True):
        txt = el.get_text(separator=" ", strip=True)
        if txt and len(txt) > 2:
            chunks.append(txt)

    text = "\n\n".join(chunks)

    for pattern in NOISE_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line for line in text.split("\n") if line.strip()]
    text = "\n".join(lines)
    text = re.sub(r"\n\n+", "\n\n", text)

    return text.strip()


def process_bank_lending_url(landing_page_url, title_from_csv):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    try:
        response = requests.get(landing_page_url, headers=headers, timeout=15)
        soup = BeautifulSoup(response.content, "html.parser")

        meta = extract_metadata(soup)
        body_text = extract_clean_lending_text(soup)

        return {
            "url": landing_page_url,
            "title": meta["title"] if meta["title"] != "Unknown" else title_from_csv,
            "category": "Bank Lending Survey",
            "exact_date": meta["date"],
            "full_text_raw": body_text,
        }

    except Exception as e:
        return {
            "url": landing_page_url,
            "title": title_from_csv,
            "category": "Bank Lending Survey",
            "exact_date": "Error",
            "full_text_raw": f"Error: {str(e)}",
        }


def scrape_bank_lending_web_reports():
    output_dir = os.path.join("02_Data", "raw_text")
    input_path = os.path.join("02_Data", "urls", "urls_pubs_bank_lending_survey.csv")

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)

    df["year_int"] = df["date"].str.extract(r"(\d{4})").astype(int)
    df_web = df[df["year_int"] >= 2020].copy()

    print(f"total_urls={len(df)}")
    print(f"web_candidates={len(df_web)}")

    all_data = []
    successful = 0
    failed = 0

    for _, row in tqdm(df_web.iterrows(), total=len(df_web), desc="Extracting"):
        result = process_bank_lending_url(row["url"], row["title"])
        all_data.append(result)

        if result["exact_date"] != "Error":
            successful += 1
        else:
            failed += 1
            print(f"\nWARNING: Failed: {row['title']}")

        time.sleep(1)

    if not all_data:
        print("No data extracted.")
        return

    output_df = pd.DataFrame(all_data)

    threshold_date = "2020-04-01"
    filtered_rows = []
    skipped = 0

    for _, row in output_df.iterrows():
        if row["exact_date"] in ["Unknown", "Error"]:
            filtered_rows.append(row)
            continue
        try:
            if row["exact_date"] >= threshold_date:
                filtered_rows.append(row)
            else:
                skipped += 1
        except Exception:
            filtered_rows.append(row)

    final_df = pd.DataFrame(filtered_rows)
    output_path = os.path.join(output_dir, "raw_text_bank_lending_survey.csv")

    final_df.to_csv(
        output_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        encoding="utf-8-sig",
        lineterminator="\n",
    )

    print(f"\nsaved: {output_path}")
    print(f"total_reports={len(final_df)}")
    print(f"successful={successful}")
    print(f"failed={failed}")
    if skipped:
        print(f"skipped_before_{threshold_date}={skipped}")

    valid_dates = final_df[final_df["exact_date"] != "Error"]["exact_date"]
    if len(valid_dates) > 0:
        print(f"date_range={valid_dates.min()} to {valid_dates.max()}")


if __name__ == "__main__":
    scrape_bank_lending_web_reports()