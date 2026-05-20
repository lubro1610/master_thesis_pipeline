# In reality, the logic behind this cleaning script is identical to the one seen in 01_clean_mpr_pdf.py, used for cleaning mpr reports.
# The only real difference is the structure of the noise filters/regex patterns, as I tried to "tailor" them to the elements I found in each report.
# So, please see 01_clean_mpr_pdf.py for a more detailed walkthrough of the logic behind the cleaning step, and why its done for just the mprs and finstab reports.

import pandas as pd
import os
import re

# --- AGGRESSIVE NOISE FILTERS FOR FINANCIAL STABILITY PDFs ---
PDF_NOISE_FILTERS = [
    # Metadata & publication info
    r"^Address:\s*[^\n]+$",
    r"^Postal address:\s*[^\n]+$",
    r"^E-mail:\s*[^\n]+$",
    r"^Website:\s*[^\n]+$",
    r"^Editor:\s*[^\n]+$",
    r"^Cover and design:\s*[^\n]+$",
    r"^Printing:\s*[^\n]+$",
    r"^Design:\s*[^\n]+$",
    r"^ISSN\s+\d+.*$",
    r"The text is set in\s+[^\n]+$",
    r"Bankplassen\s+\d+[^\n]*",
    r"Postboks\s+\d+[^\n]*",
    
    # Page numbers (only digits on a line)
    r"^\s*\d{1,3}\s*$",
    
    # Header/Footer patterns
    r"FINANCIAL STABILITY.*?\d+\s*$",
    r"^(?:FIRST|SECOND|THIRD|FOURTH)\s+HALF\s+\d{4}",
    r"Cut-off date.*?:\s*\d+[^\n]*",
    
    # Table of Contents and Boxes - entire sections
    r"(?:^|\n)\s*(?:Table of )?Contents:?[^\n]*(?:\n(?:\s*-{1,}\s*|--?\s+[^\n]+)*)*",
    r"(?:^|\n)\s*Boxes:?[^\n]*(?:\n(?:\s*-{1,}\s*|--?\s+[^\n]+)*)*",
    # Contents headers and dot-leader TOC lines
    r"^\s*Contents\s*$",
    r"^\s*Table\s+of\s+contents\s*$",
    r"^\s*\d+\.?\s+[A-Z][^\n]*\.{2,}\s*\d+\s*$",
    r"^\s*\d+(?:\.\d+)+\s+[A-Z][^\n]*\s+\d+\s*$",
    
    # Boilerplate text
    r"The Report is published.*?(?:next Report|next Financial Stability Report)[^\n]*",
    r"At its meeting.*?(?:Executive Board|management)[^\n]*",
    
    # Appendix/Annex
    r"Appendix:?[^\n]*\n",
    r"Annex:?[^\n]*\n",
    
    # Repeated headers
    r"(?:Financial Stability Report|Financial Stability)\s+(?:\d+/)?(?:\d{4}|\d{2})",
    
    # Chart axis labels
    r"^\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[−\-]?\d{2}\s*$",
    
    # Footnote reference markers
    r"^\s*\d+\)\s*",
    r"^\s*\d+\s*\)\s*$",
    
    # Chart references
    r"\(Chart\s+\d+\.?\d*\)",
    r"^\s*Chart\s+\d+\.?\d*[^\n]*$",
    
    # Sources lines
    r"^\s*Sources?:\s*.*$",
    r"^\s*Source\s*[:\-].*$",
    r"^.*Sources?:\s+.*$",
    
    # Common institution names in table contexts
    r"^\s*(?:OECD|IMF|ECB|Fed|EU Commission|Eurostat|World Bank)\s*\d*\)?\s*$",
    r"^\s*(?:Statistics Norway|Norges Bank|Ministry of Finance)\s*\d*\)?\s*$",
    r"^\s*Private institutions?\s*\d*\)?\s*$",
    r"^\s*Trading partners?\s*\d*\)?\s*$",
    
    # Table estimate/statistic labels
    r"^\s*(?:Highest|Lowest|Average|Median)\s+estimate\s*$",
    r"^\s*(?:Highest|Lowest|Average|Median)\s*$",
    
    # Footnote descriptions
    r"^\s*\d+\)\s+[A-Z].*$",
    r"^\s*\d+\)\s+[a-z].*$",
    
    # Table headers
    r"^\s*TABLE\s+\d+.*$",
    r"^\s*Table\s+\d+.*$",
    r"^\s*[Tt]able\s+\d+\s*[:\.]?\s+.*$",
    r"^\s*(?:Change from|Percent|Percentage|Share of|Level|Index|Rate|Growth).*$",
    r"^\s*(?:projections?|estimates?|brackets?|figures?).*$",
    
    # Chart date ranges
    r"^\s*(?:Percent|Index|Per cent)\.\s+(?:Quarterly|Monthly|Annual)\s+figures\.\s+\d{4}.*\d{4}.*$",
    r"^\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*[–-]\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)?\s*\d{4}.*$",
    
    # Superscript notation cleanup
    r"[¹²³⁴⁵⁶⁷⁸⁹⁰⁾]",
    
    # URLs
    r"^\s*(?:https?://|www\.)[^\s]+.*$",
    
    # Table of contents entries
    r"^\s*\d+\.\s+[A-Z][a-z].*$",
]

def is_purely_numeric_line(line):
    clean = line.strip()
    if not clean:
        return True
    for char in clean:
        if not (char.isdigit() or char in ' .()-'):
            return False
    return any(c.isdigit() for c in clean)

def is_mostly_numbers_and_punctuation(line, threshold=0.75):
    if not line.strip():
        return True
    clean = line.strip()
    allowed = sum(1 for c in clean if c.isdigit() or c in '().,% −-')
    total = len(clean)
    return (allowed / total) >= threshold

def is_incomplete_line(line, max_length=20):
    clean = line.strip()
    if not clean:
        return True
    if clean.endswith('-'):
        return False
    if clean[0].islower():
        return False
    if len(clean) <= max_length:
        if not clean.endswith(('.', '?', '!', ':', ';', ')', '"')):
            return True
    return False

def clean_pdf_text(raw_text):
    """Remove known noise from Financial Stability PDF text."""
    text = raw_text
    for pattern in PDF_NOISE_FILTERS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove table lines (mostly numeric) and incomplete headers
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        if is_purely_numeric_line(line):
            continue
        if not is_mostly_numbers_and_punctuation(line, threshold=0.75) and not is_incomplete_line(line, max_length=20):
            filtered_lines.append(line)
    text = '\n'.join(filtered_lines)
    
    # Normalize whitespace
    text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\n{2,}', '\n\n', text)
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    text = '\n'.join(lines)
    return text.strip()

def run_pdf_cleaning(input_csv, output_csv=None):
    if output_csv is None:
        output_csv = input_csv.replace('_pdf.csv', '_pdf_CLEAN.csv')
    
    df = pd.read_csv(input_csv)
    
    print(f"documents={len(df)}")
    df['full_text_clean'] = df['full_text_raw'].apply(clean_pdf_text)
    
    # Summary statistics
    raw_chars = df['full_text_raw'].astype(str).str.len().sum()
    clean_chars = df['full_text_clean'].astype(str).str.len().sum()
    reduction = (1 - clean_chars / raw_chars) * 100 if raw_chars > 0 else 0
    print(f"raw_chars={raw_chars:,}")
    print(f"clean_chars={clean_chars:,}")
    print(f"noise_removed={reduction:.1f}%")
    
    output_path = os.path.join("02_Data", "raw_text", os.path.basename(output_csv))
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"saved: {output_path}")
    
    return df

if __name__ == "__main__":
    input_file = os.path.join("02_Data", "raw_text", "raw_text_financial_stability_pdf.csv")
    run_pdf_cleaning(input_file)
