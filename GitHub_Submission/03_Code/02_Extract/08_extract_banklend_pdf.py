# Package structure is pretty much identical throughout all the extraction scripts, as the logic is inherently similar.
# Please look to 01_extract_speeches.py for details on the overall package structure and the main extraction logic.
# You may find artifacts across the extraction scripts (such as logic for speaker extraction etc, even though only speeches have speakers)
# This was a slight shortcut to streamline the process, but it does not affect the final output in any way, as the logic is only applied if relevant. 
# 04_extract_mpr_pdf.py contains some notes on the extraction logic of the older (after Q3 2021) PDF reports.

import csv
import io
import os
import re
import time

import fitz  # PyMuPDF
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin

# PDF-specific filters
PDF_NOISE = [
    r"^\d+\s*Survey of Bank Lending.*?\d+",     # Page headers
    r"Norges Bank\s*\|.*?\|",                 # Footers
    r"Source:\s+.*",                          # Source citations
    r"^\d+$",                                 # Pure page numbers
    r"^\s*-{3,}\s*$",                         # Separator lines
    r"ISSN\s+\d{4}-\d{4}",                    # ISSN numbers
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
            if len(doc) > 10:
                start_page = 2
                end_page = len(doc) - 2
            else:
                start_page = 0
                end_page = len(doc)

            for page_num in range(start_page, end_page):
                page = doc[page_num]
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: (b[1], b[0]))

                for b in blocks:
                    block_text = b[4].strip()

                    if len(block_text) < 15:
                        continue
                    if is_table_dense(block_text):
                        continue

                    for pattern in PDF_NOISE:
                        block_text = re.sub(pattern, "", block_text, flags=re.IGNORECASE | re.MULTILINE)

                    if block_text.strip():
                        full_text.append(block_text.strip())

        return "\n\n".join(full_text)
    except Exception as e:
        return f"PDF_ERROR: {e}"


def clean_bank_lending_text(text):
    """Remove recurring boilerplate and chart/image sections from PDF text."""
    if not text:
        return text
    # I discovered that bank lending surveys, especially the older ones, have a lot of reoccuring boilerplate elements that are not relevant.
    # There are some examples that I found in many of the reports, and consequently created some regex patterns to remove them.
    boilerplate_patterns = [
        r"Norges Bank’s quarterly bank lending survey is a qualitative survey.*?2007 Q4\.?",
        r"The questions distinguish between lending to households and lending to non-financial enterprises.*?loan volume for the past couple of years\.?",
        r"Some changes have been made to the questions in the survey.*?Survey of Bank Lending.*?lending survey\.?",
        r"Some changes have been made to the questions in the survey and the way the results are reported.*?lending\s+survey\.?",
        r"The banks in the survey are asked to assess developments.*?will be\s*[‐-]?\s*100%?\.?",
        r"(?:In the survey, there is|The banks in the survey use) a scale of five alternative\s+responses.*?(?:demand, the net balance will be 100|demand will be 100|the net\s+balance for demand will be 50.*?balance will be 100)\.?",        r"The banks use a scale of five alternative responses to indicate the degree of change in credit\s+conditions.*?the net\s+percentage balance will be\s*[‐-]?\s*100\s*%\.?",    ]

    cleaned = text
    for pattern in boilerplate_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE | re.DOTALL)

    # As can be seen here I also stepped up the regex cleaning of other elements as I went. 
    # These are quite aggressive filters, but it is intentional to remove reoccuring elements.
    # Use '^' and '$' to make sure we only target exact matches at the start and end of lines.
    chart_line_patterns = [
        r"^Chart\s+\d+",
        r"^Change from previous quarter",
        r"^Next quarter$",
        r"^Households$",
        r"^Non-financial enterprises$",
        r"^Residential mortgages\.$",
        r"^Total credit to non-financial enterprises\.$",
        r"^2/1\s*=",
        r"^0\s*=",
        r"^\d+\s*Residential mortgages\.$",
        r"^\d+\s*Total credit to non-financial enterprises\.$",
        r"^\d+\s*2/1\s*=",
        r"^\d+\s*As\s+an\s+increase",
        r"^Red dots show expected developments",
        r"^Blue bars show reported developments",
        r"^The blue line shows reported developments",
        r"^The blue dot shows expected developments",
        r"^Aggregate demand",
        r"^Overall credit standards",
        r"^Factors that have contributed to the changes",
        r"^Selected loan types$",
        r"^First-home mortgages$",
        r"^Fixed-rate mortgages$",
        r"^Commercial real estate$",
        r"^Credit line utilisation rate$",
        r"^Market share objectives$",
        r"^Economic outlook$",
        r"^Sector-specific outlook$",
        r"^Banks'?\s+funding$",
        r"^risk appetite$",
        r"^Capital adequacy$",
        r"^Collateral requirements",
        r"^Equity capital requirements",
        r"^Maximum\s+DTI$",
        r"^Maximum\s+LTV$",
        r"^Maximum\s+loan\s+maturity$",
        r"^Use of interest-only periods$",
        r"^Fees\b",
        r"^If all banks responded",
        r"^the fee series has been negativised\.?$",
        r"^-1/-2\s*=",
        r"^\(the latter is the largest component\)\.?$",
    ]

    keep_short_lines = {
        "Lending to households",
        "Lending to non-financial enterprises",
    }

    def is_chart_noise(line):
        stripped = line.strip()
        if not stripped:
            return True
        if any(re.match(pat, stripped, flags=re.IGNORECASE) for pat in chart_line_patterns):
            return True
        if re.match(r"^[¹²³\d]+\s+", stripped):
            return True
        if stripped in keep_short_lines:
            return False
        # Drop short, mostly numeric lines (axis labels, ticks)
        digits = sum(c.isdigit() for c in stripped)
        letters = sum(c.isalpha() for c in stripped)
        if letters == 0 and digits > 0:
            return True
        if len(stripped) <= 3:
            return True
        if digits > 0 and (digits / max(len(stripped), 1)) > 0.35 and letters < 5:
            return True
        # Drop short label-like lines with no punctuation
        words = stripped.split()
        if len(words) <= 3 and not re.search(r"[\.,;:!?]", stripped):
            return True
        return False

    lines = [ln.strip() for ln in cleaned.splitlines()]
    cleaned_lines = [ln for ln in lines if ln and not is_chart_noise(ln)]
    return "\n".join(cleaned_lines).strip()

def extract_metadata(soup):
    """Extract date from landing page metadata."""
    metadata = {"date": "Unknown"}

    date_tag = soup.find("meta", attrs={"property": "article:published_time"})
    if date_tag:
        date_content = date_tag.get("content", "")
        date_part = date_content.split()[0] if " " in date_content else date_content
        try:
            from datetime import datetime
            dt = datetime.strptime(date_part, "%m/%d/%Y")
            metadata["date"] = dt.strftime("%Y-%m-%d")
        except Exception:
            metadata["date"] = date_part

    return metadata


def find_pdf_link(soup):
    """Finds PDF link using multiple patterns for different years."""
    pdf_url = None

    # Pattern 1: Enhanced download link
    download_link = soup.find("p", class_=re.compile(r"download-link", re.I))
    if download_link:
        pdf_link = download_link.find("a", href=re.compile(r"\.pdf", re.IGNORECASE))
        if pdf_link:
            pdf_url = pdf_link.get("href")

    # Pattern 2: Any PDF link in downloads section
    if not pdf_url:
        downloads = soup.find(["section", "div"], class_=re.compile(r"download", re.I))
        if downloads:
            pdf_link = downloads.find("a", href=re.compile(r"\.pdf", re.IGNORECASE))
            if pdf_link:
                pdf_url = pdf_link.get("href")

    # Pattern 3: Generic fallback
    if not pdf_url:
        pdf_link = soup.find("a", href=re.compile(r"\.pdf", re.IGNORECASE))
        if pdf_link:
            pdf_url = pdf_link.get("href")

    return pdf_url


def process_landing_page(landing_url, headers):
    """Finds and fetches PDF from landing page, extracts metadata and text."""
    try:
        res = requests.get(landing_url, headers=headers, timeout=15)
        soup = BeautifulSoup(res.content, "html.parser")

        metadata = extract_metadata(soup)
        pdf_url = find_pdf_link(soup)

        if pdf_url:
            pdf_url = urljoin("https://www.norges-bank.no", pdf_url)
            pdf_res = requests.get(pdf_url, timeout=20)
            text = extract_pdf_blocks(pdf_res.content)
            text = clean_bank_lending_text(text)

            return {
                "text": text,
                "date": metadata["date"],
            }

        return {
            "text": "NO_PDF_LINK_FOUND",
            "date": "Unknown",
        }

    except Exception as e:
        return {
            "text": f"LANDING_PAGE_ERROR: {e}",
            "date": "Unknown",
        }


def extract_year_from_date(date_str):
    match = re.search(r"(\d{4})", str(date_str))
    if match:
        return int(match.group(1))
    return None


def run_bank_lending_pdf_scrape(csv_path):
    df_urls = pd.read_csv(csv_path)

    df_urls["year"] = df_urls["date"].apply(extract_year_from_date)
    df_legacy = df_urls[df_urls["year"] <= 2019].copy()

    print(f"total_reports={len(df_urls)}")
    print(f"pdf_reports={len(df_legacy)}")
    print(f"web_reports={len(df_urls) - len(df_legacy)}")

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    results = []
    successful = 0
    failed = 0

    for _, row in tqdm(df_legacy.iterrows(), total=len(df_legacy), desc="Extracting"):
        result = process_landing_page(row["url"], headers)

        text = result["text"]
        exact_date = result["date"]

        if text and not text.startswith(("NO_PDF_LINK", "LANDING_PAGE_ERROR", "PDF_ERROR")):
            successful += 1
        else:
            failed += 1

        results.append(
            {
                "url": row["url"],
                "title": row["title"],
                "category": "Bank Lending Survey",
                "exact_date": exact_date,
                "full_text_raw": text,
            }
        )

        time.sleep(1.5)

    if results:
        df_out = pd.DataFrame(results)
        output_path = os.path.join("02_Data", "raw_text", "raw_text_bank_lending_survey_pdf.csv")

        df_out.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8-sig", lineterminator="\n")

        print(f"\nsaved: {output_path}")
        print(f"total_reports={len(df_out)}")
        print(f"successful={successful}")
        print(f"failed={failed}")

        successful_texts = [
            r["full_text_raw"]
            for r in results
            if not r["full_text_raw"].startswith(("NO_PDF_LINK", "LANDING_PAGE_ERROR", "PDF_ERROR"))
        ]
        if successful_texts:
            lengths = [len(t) for t in successful_texts]
            print(f"text_length_min={min(lengths):,}")
            print(f"text_length_median={int(pd.Series(lengths).median()):,}")
            print(f"text_length_max={max(lengths):,}")


if __name__ == "__main__":
    csv_path = os.path.join("02_Data", "urls", "urls_pubs_bank_lending_survey.csv")
    run_bank_lending_pdf_scrape(csv_path)