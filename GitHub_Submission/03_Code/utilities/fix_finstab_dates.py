"""
Fix Missing Dates in Financial Stability Reports
=================================================

Financial Stability reports from 2018 and earlier lack exact_date values.
This script infers publication dates from:
1. Title patterns (e.g., "Financial Stability 1/06" = first half 2006)
2. URL patterns (e.g., "Financial-Stability-1-06")

Publication dates are set to:
- First half reports: May 15 (approximate spring publication)
- Second half reports: November 15 (approximate autumn publication)

Author: Master Thesis
Date: February 2026
"""

import pandas as pd
import re
from datetime import datetime

def infer_date_from_title(title, url):
    """
    Infer publication date from Financial Stability report title.
    
    Patterns:
    - "Financial Stability 1/06" -> 2006-05-15 (first half)
    - "Financial Stability 2/06" -> 2006-11-15 (second half)
    - "Financial Stability Report 2014" -> 2014-11-15 (annual report)
    
    Args:
        title: Report title
        url: Report URL (backup for parsing)
    
    Returns:
        Date string (YYYY-MM-DD) or None
    """
    if pd.isna(title):
        return None
    
    # Pattern 1: "Financial Stability 1/06" or "Financial Stability 2/12"
    match = re.search(r'Financial Stability (\d)/(\d{2})', title)
    if match:
        half = int(match.group(1))
        year_short = match.group(2)
        
        # Convert 2-digit year to 4-digit (00-99 -> 2000-2099)
        year = 2000 + int(year_short)
        
        # First half = May 15, Second half = November 15
        month = 5 if half == 1 else 11
        day = 15
        
        return f"{year}-{month:02d}-{day:02d}"
    
    # Pattern 2: "Financial Stability Report 2014" (annual reports)
    match = re.search(r'Financial Stability Report (\d{4})', title)
    if match:
        year = int(match.group(1))
        # Annual reports typically published in autumn
        return f"{year}-11-15"
    
    # Pattern 3: Try to extract from URL as fallback
    match = re.search(r'Financial-Stability-(\d)-(\d{2})', url)
    if match:
        half = int(match.group(1))
        year_short = match.group(2)
        year = 2000 + int(year_short)
        month = 5 if half == 1 else 11
        return f"{year}-{month:02d}-{day:02d}"
    
    match = re.search(r'Financial-stability-(\d{4})', url)
    if match:
        year = int(match.group(1))
        return f"{year}-11-15"
    
    return None


def main():
    """
    Fix missing dates in Financial Stability consolidated file.
    """
    print("="*70)
    print("Fixing Missing Dates in Financial Stability Reports")
    print("="*70)
    
    # Load file
    input_file = '02_Data/clean_consolidated/raw_text_financial_stability_consolidated.csv'
    df = pd.read_csv(input_file)
    
    print(f"\nLoaded {len(df)} Financial Stability reports")
    
    # Count missing dates
    missing_before = df['exact_date'].isna().sum()
    print(f"Missing dates: {missing_before}")
    
    if missing_before == 0:
        print("No missing dates found. Nothing to fix!")
        return
    
    # Infer dates for missing rows
    print("\nInferring dates from titles and URLs...")
    
    for idx, row in df.iterrows():
        if pd.isna(row['exact_date']):
            inferred_date = infer_date_from_title(row['title'], row['url'])
            
            if inferred_date:
                df.at[idx, 'exact_date'] = inferred_date
                print(f"  Row {idx}: {row['title'][:50]}... -> {inferred_date}")
            else:
                print(f"  Row {idx}: Could not infer date for: {row['title']}")
    
    # Check results
    missing_after = df['exact_date'].isna().sum()
    fixed_count = missing_before - missing_after
    
    print(f"\nFixed {fixed_count} dates")
    print(f"Remaining missing dates: {missing_after}")
    
    # Save updated file
    df.to_csv(input_file, index=False, encoding='utf-8-sig')
    print(f"\nUpdated file saved: {input_file}")
    
    # Show date distribution
    print("\nDate distribution:")
    df['year'] = pd.to_datetime(df['exact_date']).dt.year
    print(df['year'].value_counts().sort_index())
    
    print("\n" + "="*70)
    print("Date fixing complete!")
    print("="*70)


if __name__ == "__main__":
    main()
