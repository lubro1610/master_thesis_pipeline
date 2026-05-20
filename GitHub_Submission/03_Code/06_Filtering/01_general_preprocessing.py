# Takes the consolidated CSVs and cleans them up for sentiment analysis.
# Main jobs: strip metadata lines left over from extraction, standardize dates,
# filter speeches to committee members only, drop non-monetary press releases,
# and remove very short documents.
# Output goes to 02_Data/preprocessed/
# The docstring above provides a fairly clear overview of the role of this script and the cleaning steps that this script performs.
# In terms of the imports, as per usual, we use pandas for data manipulation, re for cleaning and normalizing whitespace, and pathlib for file path handling.
import pandas as pd
import re
from pathlib import Path

# So this function is first and foremost used to remove metadata-lines that I added during the extraction/consolidation process prior to proper CSV standardization,
# just to keep track of things. However, it also has some useful steps in terms of normalizing whitespace that occurs due to the cleaning process, as 
# filtered lines can leave excessive line breaks and spaces. This is purely cosmetic though. 
def clean_text(text):
    # Strips out the metadata lines I added during extraction (SOURCE:, TITLE: etc)
    # and normalizes whitespace left behind from the cleanup.
    if pd.isna(text):
        return ""
    
    # Remove metadata lines
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line_upper = line.strip().upper()
        
        # Skip metadata lines
        if any(line_upper.startswith(prefix) for prefix in [
            'SOURCE:',
            'TITLE:',
            'SPEAKER:',
            'DATE:',
            'URL:',
            '---',
            '==='
        ]):
            continue
        
        # Skip empty lines
        if not line.strip():
            continue
        
        cleaned_lines.append(line.strip())
    
    # Join and normalize whitespace
    cleaned = ' '.join(cleaned_lines)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()

# If you've read my other comments throughout the codebase, youll have noticed that I made kind of a mess out of the date formats
# and shown very little consistency in that regard. Though immaterial in terms of the logic, its nice to finally have a standardized format
# to work with. So this function takes care of converting various formats to a uniform YYYY-MM-DD format.
def standardize_date(date_str):
    # The date formats across documents were all over the place, so this just
    # converts everything to YYYY-MM-DD.
    if pd.isna(date_str):
        return None
    
    dt = pd.to_datetime(date_str, errors='coerce')
    if pd.notna(dt):
        return dt.strftime('%Y-%m-%d')
    
    return None

# So for this function youll see that we load a whitelist of committee members from a CSV file in the Environment folder. In order to ensure
# that the speakers we integrate in the speeches dataset are governors/deputy governors and not other speakers, we filter the speeches based on a 
# curated "whitelist". The logic of that list if quite simple, and it required me to check if a given speaker served as either governor or deputy governor at 
# the time of the speech, on a strict or interim basis. If they did, they got a YES in the "include" column, and if not the speeches are disregarded.
# The script isnt very smart, and does no smart/fuzzy matching. As such, it was necessary to ensure that the names in the whitelist 
# (some including titles etc, some not) were an exact match to the speaker names in the original CSV files. Bit manual, but still straighforward and effective
# for this kind of fairly small dataset.
def load_committee_whitelist():
    # Loads the manually curated list of governors/deputy governors from Environment/.
    # Returns a set of names for exact matching - see the CSV for the inclusion logic.
    # Look for whitelist in Environment folder
    whitelist_file = Path(__file__).parent.parent.parent / 'Environment' / 'committee_members_whitelist.csv'
    
    if not whitelist_file.exists():
        print(f"  WARNING: committee_members_whitelist.csv not found in Environment/!")
        print(f"      All speakers will be included (no filtering)")
        return None
    
    df = pd.read_csv(whitelist_file, encoding='utf-8-sig')  # Handle BOM (byte order mark)if present
    
    # Debug: Check if include column has data
    if df['include'].isna().all():
        print(f"  WARNING: 'include' column is empty in whitelist!")
        print(f"      Make sure file is saved with YES/NO values")
        return None
    
    # Ensure include column is string type and handle missing values
    df['include'] = df['include'].fillna('NO').astype(str).str.strip()
    
    # Filter to only YES entries (case-insensitive)
    approved = df[df['include'].str.upper() == 'YES']['speaker'].tolist()
    
    # Convert to set for fast lookup (exact matching)
    approved_set = set(approved)
    
    return approved_set

# While I dont filter press releases on a word basis like for the other reports, its still vital that the corpus doesnt retain 
# non-monetary press releases, such as the issuing of a commemorative coin or some FX transaction that simply isnt relevant for the analysis.
# As such, a list of exclusion patterns is defined in this function, and any release with title matching these patterns is removed.
# Obviously, its a bit blunt, and probably not 100% effective, so Ill be sure to do a manual check afterwards.
def get_press_release_exclusion_patterns():
    # Title patterns for press releases we want to drop - FX transactions, coins,
    # property investments, staff appointments and similar non-monetary content.
    return [
        # FX transactions (monthly technical announcements)
        "foreign exchange purchases for the Government Pension Fund",
        "Foreign exchange to the Government Pension Fund Global",
        "foreign exchange transactions in",  # catches "in January 2025" etc
        "Norges Bank's foreign exchange purchases",
        "Norges Bank's foreign exchange transactions",
        
        # Commemorative coins (cultural, not monetary policy)
        "commemorative coin",
        
        # NBIM property investments (not monetary policy)
        "Fund makes",
        "Fund enters agreement",
        "property investment",
        "Fund to make first property investment",
        "First property investment",
        
        # Banknotes/coins technical
        "Old banknotes no longer legal tender",
        "50-øre coin retires",
        "Redemption of old",
        "notes printed abroad put into circulation",
        
        # Staff appointments (not monetary policy communication)
        "appointed Deputy Governor",
        "appointed CEO",
        "appointed Director",
        "appointed new Deputy Governor",
        
        # Administrative/technical
        "Changes to the rules on collateral",
        "Changes to the guidelines for collateral",
        "Changes in the guidelines for pledging collateral",
        "Temporary changes in the Guidelines for pledging"
    ]

# The main function of the script, run five times, so one fo each document cateogry. Docstring provides a fairly brief overview.
def process_file(input_file, output_file, min_words=50, start_year=2005,
                 committee_whitelist=None):
    # Runs all cleaning steps on one file. start_year=2005 because we need Dec 2005
    # documents for the first 2006 meeting window.
    print(f"\n{input_file}")
    
    # Load data as the consolidated CSVs for each category.
    df = pd.read_csv(f'02_Data/clean_consolidated/{input_file}')
    print(f"documents_in={len(df)}")
    
    # Filter speeches by committee whitelist (BEFORE any other processing)
    if 'speech' in input_file.lower() and committee_whitelist is not None:
        if 'speaker' in df.columns:
            # Count before
            before_count = len(df)
            
            # EXACT matching - speaker name must be in whitelist exactly
            # This ensures "Executive Director Ida Wolden Bache" is excluded
            # even though "Ida Wolden Bache" is in the whitelist.
            # Remember, we want to include only governors and deputy governors, 
            # and not the same people serving other roles at the time.
            df = df[df['speaker'].isin(committee_whitelist)].copy()
            
            # Count after
            after_count = len(df)
            excluded_count = before_count - after_count
            
            print(f"speeches_kept={after_count}")
            print(f"speeches_excluded={excluded_count}")
            
            if after_count == 0:
                print(f"  ERROR: No speeches left after filtering!")
                return None
        else:
            print(f"  WARNING: No 'speaker' column found, cannot filter by committee")
    
    # Here we apply the get_press_releases_exclusion_patterns function to press releases
    if 'press' in input_file.lower():
        if 'title' in df.columns:
            # Count before
            before_count = len(df)
            
            # Get exclusion patterns
            exclusion_patterns = get_press_release_exclusion_patterns()
            
            # Create mask for rows to KEEP (inverse of exclusion)
            keep_mask = pd.Series([True] * len(df), index=df.index)
            
            for pattern in exclusion_patterns:
                # Mark rows matching any pattern for exclusion
                matches = df['title'].str.contains(pattern, case=False, na=False)
                keep_mask = keep_mask & ~matches
            
            # Apply filter
            df = df[keep_mask].copy()
            
            # Count after
            after_count = len(df)
            excluded_count = before_count - after_count
            
            print(f"press_releases_kept={after_count}")
            print(f"press_releases_excluded={excluded_count}")
            
            if after_count == 0:
                print(f"  ERROR: No press releases left after filtering!")
                return None
        else:
            print(f"  WARNING: No 'title' column found, cannot filter press releases")
    
    # Find text column, iterates through until it finds "text" or "full" as we standardized the text column for all docs to "full_text_raw".
    # Bit of inconsistency as its not technically raw, but bear with me...
    text_col = None
    for col in df.columns:
        if 'text' in col.lower() or 'full' in col.lower():
            text_col = col
            break
    
    if not text_col:
        print(f"  ERROR: No text column found!")
        return None
    
    # Find date column -> again, all documents now have 'exact_date', so this is just an artifact from the past.
    date_col = 'exact_date' if 'exact_date' in df.columns else 'date'
    
    # Clean text
    df['text'] = df[text_col].apply(clean_text)
    
    # Standardize dates
    df['date'] = df[date_col].apply(standardize_date)
    
    # Remove rows with invalid dates
    before = len(df)
    df = df[df['date'].notna()].copy()
    if before > len(df):
        print(f"  Removed {before - len(df)} documents with invalid dates")
    
    # Filter by year. Here 'start_year' is set to 2005 by befault, not 2006, and you might be thinking "why 2005?"
    # I might be confusing you a bit, as depending on the extent to which you have read the thesis and the codebase,
    # you might be under the impression that the analysis runs from 2006 to 2025, which technically it does, but in order
    # to have sentiment values for the very first meeting in 2026, we have to include documents from December 2005,
    # so the inter-meeting period intercepting end of 2005/beginning of 2006. This logic ensures we get those documents in
    # so we can actually have sentiment values for all our meetings.
    df['year'] = pd.to_datetime(df['date']).dt.year
    before = len(df)
    df = df[df['year'] >= start_year].copy()
    if before > len(df):
        print(f"  Removed {before - len(df)} documents before {start_year}")
    
    # Remove short documents as they are unlikely to be policy-relevant when min_words is set to 50.
    # The threshold varies by document type, and main() defines these thresholds.
    df['word_count'] = df['text'].str.split().str.len()
    before = len(df)
    df_filtered = df[df['word_count'] >= min_words].copy()
    if before > len(df_filtered):
        print(f"  Removed {before - len(df_filtered)} documents with < {min_words} words")
    
    # Remove duplicates by title+date, just in case
    if 'title' in df_filtered.columns:
        before = len(df_filtered)
        df_filtered = df_filtered.drop_duplicates(subset=['title', 'date'], keep='first')
        if before > len(df_filtered):
            print(f"  Removed {before - len(df_filtered)} duplicate documents")
    
    # Select final columns
    final_columns = ['date', 'text', 'title', 'category'] if 'title' in df_filtered.columns else ['date', 'text', 'category']
    if 'category' not in df_filtered.columns:
        final_columns = [c for c in final_columns if c != 'category']
    
    df_final = df_filtered[final_columns].copy()
    
    # Sort by date
    df_final = df_final.sort_values('date').reset_index(drop=True)
    
    # Save
    output_path = f'02_Data/preprocessed/{output_file}'
    
    # Show word count stats before dropping column
    word_counts = df_filtered['word_count']
    print(f"word_count_min={word_counts.min()}, median={word_counts.median():.0f}, max={word_counts.max()}")
    
    df_final.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"saved: {output_file}")
    print(f"documents_out={len(df_final)}")
    print(f"date_range={df_final['date'].min()} to {df_final['date'].max()}")
    
    return df_final


def main():
    # Load committee whitelist for speeches filtering
    committee_whitelist = load_committee_whitelist()
    
    if committee_whitelist:
        print(f"committee_members={len(committee_whitelist)}")
    else:
        print("committee_members=all")
    
    # Define file mappings. Here youll see the thresholds put in place for the min >= word count filter.
    # As you can see, I put the threhold at 30 for press releases, 50 for speeches and bank lending surveys, 
    # and 100 for the longer reports. These thresholds are somewhat arbitrary, but very conservative at any rate.
    # If real policy-relevant entries, these threholds will not exclude any documents. 
    files = [
        ('raw_text_speeches.csv', 'preprocessed_speeches.csv', 50),
        ('raw_text_press_releases.csv', 'preprocessed_press_releases.csv', 30),  # Lower threshold for press releases
        ('raw_text_mpr_consolidated.csv', 'preprocessed_mpr.csv', 100),  # Higher for long reports
        ('raw_text_financial_stability_consolidated.csv', 'preprocessed_finstab.csv', 100),
        ('raw_text_bank_lending_survey_consolidated.csv', 'preprocessed_banklend.csv', 50)
    ]
    
    results = {}
    
    for input_file, output_file, min_words in files:
        df = process_file(input_file, output_file, min_words, 
                         committee_whitelist=committee_whitelist)
        if df is not None:
            results[output_file] = df
    
    total_docs = 0
    for output_file, df in results.items():
        total_docs += len(df)
    print(f"\ntotal_documents={total_docs}")


if __name__ == "__main__":
    main()
