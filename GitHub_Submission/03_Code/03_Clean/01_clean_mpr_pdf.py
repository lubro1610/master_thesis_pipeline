# So for mprs and financial stability reports, I implemented two cleaning scripts. The reason why speeches, press releases and bank lending surveys
# dont have cleaning scripts is that I found the initial extraction to be sufficient for those categories. Press releases and speeches
# are all web-based, and surveys of bank lending are quite short. While they do have some noise, I ramped up the regex patterns for those reports,
# as I found that the lower amount of regex filters in the other reports could use some work. The result was a fairly clean corpus for the bank lending surveys.
# Far from perfect, as working with PDFs is always difficult, but it was sufficient to the extent that I wouldnt need a separate cleaning script.

# Now onto the specifics of the cleaning scripts. This script takes the raw text extracted from the PDFs, which I called legacy (even though not all of them are that old)
# The idea is to really home in on the specific noise elements, and try to remove everything that is not narrative text.
# Elements such as tabular data, headers, footers, references to tables and charts etc. are all elements that could distrort the narrative flow, and later on our sentiment analysis.


import pandas as pd # pandas for readign and writing CSVs
import os # os to build file paths
import re # re for regex cleaning of text

# --- AGGRESSIVE NOISE FILTERS FOR LEGACY PDFs ---
# This is really just one big list of regex patterns, where I tried to target the typical noise elements I found to be reoccuring through the reports.
# I didnt have much experience with this beforehand, but I use re.sub() to replace any matches with an empty string, effectively removing the identified noise from the text.
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
    
    # Page numbers (digits only on a line)
    r"^\s*\d{1,3}\s*$",
    
    # Header/footer patterns from legacy PDFs
    r"ECONOMIC BULLETIN.*?\d+\s*$",
    r"^(?:FIRST|SECOND|THIRD|FOURTH)\s+QUARTER\s+\d{4}",
    r"Cut-off date.*?:\s*\d+[^\n]*",
    
    # Table of Contents and Boxes - entire sections
    r"(?:^|\n)\s*(?:Table of )?Contents:?[^\n]*(?:\n(?:\s*-{1,}\s*|--?\s+[^\n]+)*)*",
    r"(?:^|\n)\s*Boxes:?[^\n]*(?:\n(?:\s*-{1,}\s*|--?\s+[^\n]+)*)*",
    
    # Boilerplate text about report publication
    r"The Report is published.*?(?:next Report|next Inflation Report)[^\n]*",
    r"At its meeting.*?(?:Executive Board|management)[^\n]*",
    
    # Appendix / annex sections
    r"Appendix:?[^\n]*\n",
    r"Annex:?[^\n]*\n",
    
    # Spaced-out text from older PDF generations
    r"I n f l a t i o n\s+R e p o r t\s+\d+/\d+",
    r"M o n e t a r y\s+P o l i c y\s+R e p o r t",
    
    # Repeated headers across pages
    r"(?:Inflation Report|Monetary Policy Report)\s+(?:\d+/)?(?:\d{4}|\d{2})",
    
    # --- CHART AXIS LABELS & REFERENCES ---
    # Month-year labels from chart axes (Jan−14, Jul−15, etc.)
    r"^\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[−\-]?\d{2}\s*$",
    
    # Footnote reference markers (1), 2), etc.)
    r"^\s*\d+\)\s*",
    r"^\s*\d+\s*\)\s*$",
    
    # Chart references in parentheses (Chart 1.12, Chart 2.32)
    r"\(Chart\s+\d+\.?\d*\)",
    r"^\s*Chart\s+\d+\.?\d*[^\n]*$",
    
    # --- SOURCES & INSTITUTIONAL METADATA ---
    # Sources lines (common in charts/tables)
    r"^\s*Sources?:\s*.*$",
    r"^\s*Source\s*[:\-].*$",
    r"^.*Sources?:\s+.*$",  # Sources anywhere in line
    
    # Institution names in table contexts
    r"^\s*(?:OECD|IMF|ECU|ECB|Fed|EU Commission|Eurostat|World Bank)\s*\d*\)?\s*$",
    r"^\s*(?:Statistics Norway|Norges Bank|Ministry of Finance)\s*\d*\)?\s*$",
    r"^\s*Private institutions?\s*\d*\)?\s*$",
    r"^\s*Trading partners?\s*\d*\)?\s*$",
    
    # Table estimate/statistic labels
    r"^\s*(?:Highest|Lowest|Average|Median)\s+estimate\s*$",
    r"^\s*(?:Highest|Lowest|Average|Median)\s*$",
    
    # Footnote descriptions (lines starting with 1), 2), etc. followed by text)
    r"^\s*\d+\)\s+[A-Z].*$",
    r"^\s*\d+\)\s+[a-z].*$",
    
    # --- CHART METADATA & TABLE HEADERS ---
    # Chart axis labels in text
    r"\((?:left|right)-hand scale\)",
    r"^\s*(?:left|right)-hand scale\s*$",
    
    # Table headers and titles
    r"^\s*TabLE\s+\d+.*$",
    r"^\s*TABLE\s+\d+.*$",
    r"^\s*Table\s+\d+.*$",
    r"^\s*[Tt]able\s+\d+\s*[:\.]?\s+.*$",  # Table + number + optional text
    
    # Table column headers with common patterns
    r"^\s*(?:Change from|Percent|Percentage|Share of|Level|Index|Rate|Growth).*$",
    r"^\s*(?:projections?|estimates?|brackets?|figures?).*$",
    
    # Chart date ranges (always metadata)
    r"^\s*(?:Percent|Index|Per cent)\.\s+(?:Quarterly|Monthly|Annual)\s+figures\.\s+\d{4}.*\d{4}.*$",
    r"^\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*[–-]\s*(?:January|February|March|April|May|June|July|August|September|October|November|December)?\s*\d{4}.*$",
    
    # Superscript notation cleanup
    r"[¹²³⁴⁵⁶⁷⁸⁹⁰⁾]",
    
    # --- APPENDIX & CONTACT LISTS ---
    # Remove entire contact list sections (handle hyphenated words)
    r"Enterprises and organisations.*?contac.*?work on this.*?(?=\n\n[A-Z]|Annex|$)",
    
    # Remove "Annex" sections entirely
    r"^Annex\s+[IVX]+.*?(?=\n\n|\Z)",
    
    # Company names on isolated lines (ending with AS, BA, ASA, kommune, etc.)
    r"^\s*[A-ZÆØÅ][A-Za-zæøå\s&\-\.0-9]+(?:AS|BA|ASA|kommune|AS\s|BA\s)\s*$",
    
    # Isolated country/city names (single words, capitalized, on own line)
    r"^\s*(?:Denmark|Sweden|Finland|Norway|Germany|France|UK|USA|Canada|Australia|Japan|China|India|Brazil|Russia|Poland|Netherlands|Belgium|Spain|Italy|Portugal|Austria|Switzerland|Toronto|Oslo|London|Stockholm|Copenhagen|Berlin|Paris|Rome|Madrid|Brussels|Amsterdam|Vienna|Zurich)\s*$",
    
    # --- ADDITIONAL TABLE & FORMATTING NOISE ---
    # Spaced-out headers (I N F L A T I O N, R E P O R T, etc.)
    r"^\s*([A-ZÆØÅ]\s+){2,}[A-ZÆØÅ]\s*$",
    
    # Spaced-out text with numbers/symbols (R e p o r t  3 / 2 0 0 5, etc.)
    r"^\s*([A-Za-zæøåÆØÅ0-9]\s+){3,}[A-Za-zæøåÆØÅ0-9/]+\s*$",
    
    # Lines with only fractions/numbers (3½, ¼, 2½, -¼, etc.) - table data
    r"^\s*[-−]?[0-9¼½¾⅓⅔⅛⅜⅝⅞]+\s*$",
    
    # Country/region names with footnote numbers (Germany1), France1), UK1), etc.)
    r"^\s*(?:Denmark|Sweden|Finland|Norway|Germany|France|UK|USA|Canada|Australia|Japan|China|India|Brazil|Russia|Poland|Netherlands|Belgium|Spain|Italy|Portugal|Austria|Switzerland|Euro\s+area|United\s+States|New\s+Zealand)\s*[¹²³⁴⁵⁶⁷⁸⁹⁰\d]+\)\s*$",
    
    # --- FOOTNOTES, URLS & TABLE OF CONTENTS ---
    # URLs (http://, https://, www.)
    r"^\s*(?:https?://|www\.)[^\s]+.*$",
    
    # Footnote explanations (lines starting with number + parenthesis + space + text)
    r"^\s*\d+\)\s+[A-Z].*$",
    
    # Table of contents entries (number + period + title, possibly with dash/bullet)
    r"^\s*\d+\.\s+[A-Z][a-z].*$",
]

# As you can probably read from the docstring below and the function name, it checks if a sentence is purely numeric, or if it contains any digit in parantheses.
# I struggled to filter any reference such as (see Table 3.2 or Chart 1.12) without also removing narrative text that contained parentheses.
# So, it is an attempt to jump that hurdl in addition to removing lines that are purely numeric such as typical table lines.
def is_purely_numeric_line(line):
    """Check if a line is 100% numeric (digits, decimals, parentheses, minus, whitespace).
    Safe to remove, never narrative text."""
    clean = line.strip()
    if not clean:
        return True
    
    # Every character must be a digit, space, period, minus, or parenthesis
    for char in clean:
        if not (char.isdigit() or char in ' .()-'):
            return False
    
    # Need at least one digit to count as numeric
    return any(c.isdigit() for c in clean)

# This function is a softer implementation than the previous, and allows a line to have some letters, but is flagged if it is >= 70% numeric.
# If you read the extraction scripts, you will see that I also tried to remove table lines already at that stage. However, while it helped it wasnt tabluar-proof.
# Anyway, the logic is that no valuable text will contain more than 70% digits, parentheses etc, and might as well be removed.
def is_mostly_numbers_and_punctuation(line, threshold=0.7):
    """Check if a line is mostly digits, parentheses, and whitespace (table data).
    Returns True when the share of numeric/punctuation chars meets the threshold."""
    if not line.strip():
        return True
    
    clean = line.strip()
    
    # Characters we'd expect in table data
    allowed = sum(1 for c in clean if c.isdigit() or c in '().,% −-')
    total = len(clean)
    
    # If >= threshold of chars are digits/punctuation, treat as table noise
    return (allowed / total) >= threshold

# So this function is the third heuristic, and it looks to identify some problems that I encountered with the initial cleaning.
# I added some max length filter in a previous implementation to try to catch noise such as headers and labels, but found that I accidentally
# removed some sentences that used hyphenation to break up long sentences. So, to avoid making the same mistake, I added some exceptions to the max length filter.
# As such, the logic that remains is short lines with less than 20 characters that do not end with typical sentence-ending punctuation are to be removed.
# This is a strong signal theyre just headers etc. In addition, any sentence starting with a lowercase letter is likely to be continuation from a previous line.
def is_incomplete_line(line, max_length=20):
    """Check if a line is short and lacks sentence-ending punctuation
    (likely a table header or isolated label)."""
    clean = line.strip()
    
    if not clean:
        return True
    
    # Keep lines ending with hyphen - they're word continuations
    if clean.endswith('-'):
        return False
    
    # Keep lines starting lowercase - probably continuation of previous line
    if clean[0].islower():
        return False
    
    # Short lines without sentence-ending punctuation are likely noise
    if len(clean) <= max_length:
        if not clean.endswith(('.', '?', '!', ':', ';', ')', '"', '\u201c', '\u201d')):
            return True
    
    return False

# This is the main function that combines all the cleaning steps. It first applies the aggressive regex filters to remove known noise elements.
# Then, it applies the three heuristics to remove lines that are purely or mostly numeric, as well as the typical headers and label elements.
def clean_pdf_text(raw_text):
    """Remove known noise from legacy MPR PDF text using regex filters and heuristics."""
    text = raw_text
    
    # Apply all regex filters
    for pattern in PDF_NOISE_FILTERS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove table lines (mostly numeric) and incomplete headers
    lines = text.split('\n')
    filtered_lines = []
    for line in lines:
        # Drop lines that are purely numeric, never narrative text
        if is_purely_numeric_line(line):
            continue
        
        # Drop table data and short orphan labels
        if not is_mostly_numbers_and_punctuation(line, threshold=0.75) and not is_incomplete_line(line, max_length=20):
            filtered_lines.append(line)
    text = '\n'.join(filtered_lines)
    
    # Normalize whitespace
    text = re.sub(r'^\s+$', '', text, flags=re.MULTILINE)
    
    # Collapse multiple blank lines into one
    text = re.sub(r'\n\s*\n{2,}', '\n\n', text)
    
    # Strip trailing whitespace per line (keep indentation for lists)
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    text = '\n'.join(lines)
    
    text = text.strip()
    
    return text

# Function that reads CSV, calls clean_pdf_ext() for every row, and saves to new CSV.
def run_pdf_cleaning(input_csv, output_csv=None):
    """Read raw PDF text from CSV, clean it, and save to a new CSV."""
    if output_csv is None:
        output_csv = input_csv.replace('_SAMPLE', '_CLEAN').replace('_legacy_pdf.', '_legacy_pdf_CLEAN.')
    
    df = pd.read_csv(input_csv)
    
    print(f"documents={len(df)}")
    
    df['Full_Text_Clean'] = df['Full_Text_Raw'].apply(clean_pdf_text)
    
    # Print summary stats
    raw_chars = df['Full_Text_Raw'].str.len().sum()
    clean_chars = df['Full_Text_Clean'].str.len().sum()
    reduction = (1 - clean_chars / raw_chars) * 100 if raw_chars > 0 else 0
    print(f"raw_chars={raw_chars:,}")
    print(f"clean_chars={clean_chars:,}")
    print(f"noise_removed={reduction:.1f}%")
    
    output_path = os.path.join("02_Data", "raw_text", os.path.basename(output_csv))
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"saved: {output_path}")
    
    return df

if __name__ == "__main__":
    # Use _SAMPLE.csv for quick test runs, full file for production
    input_file = os.path.join("02_Data", "raw_text", "raw_text_mpr_legacy_pdf.csv")
    run_pdf_cleaning(input_file)
