"""
03_merge_sentiment_to_master.py

Merges Loughran-McDonald and FinBERT sentiment scores with master dataset.
Prepares final dataset for econometric analysis with Z-score standardization.

Input:
    - 02_Data/master_dataset_2006_2025.csv (153 meetings with macro variables)
    - 02_Data/sentiment_meeting_level/lm_sentiment_meetings.csv (153 meetings)
    - 02_Data/sentiment_meeting_level/finbert_sentiment_meetings.csv (153 meetings)

Output:
    - 02_Data/master_dataset_with_sentiment.csv (153 meetings, full model)

Standardization:
    - Z-scores for LM and FinBERT net sentiment (mean=0, std=1)
    - Enables coefficient comparison: 1 unit = 1 std dev change
    - Fair "horse race" between methods

Date: 2026-02-12

"""

import pandas as pd
import numpy as np
from pathlib import Path

# File paths. Master_dataset_2006_2025.csv contains all meetings with macro variables etc, but has no sentiment.
# That is the dataset, dervied from build_master_dataset.py, which I will now merge sentiment into.
# As such, this script takes that file as input, along with the sentiment files that contain sentiment scores aggregated to the meeting level.
# The scripts that handle the sentiment scoring are found in 06_Filtering (not that intuitive, I know).
MASTER_DATA = "02_Data/master_dataset_2006_2025.csv"
LM_SENTIMENT = "02_Data/sentiment_meeting_level/lm_sentiment_meetings.csv"
FINBERT_SENTIMENT = "02_Data/sentiment_meeting_level/finbert_sentiment_meetings.csv"
OUTPUT_FILE = "02_Data/master_dataset_with_sentiment.csv"

def load_data():
    """Load all three datasets and convert dates."""
    # I load the master dataset and convert to datetime, and print meetings with data for each method to verify state of things before merging.
    master = pd.read_csv(MASTER_DATA)
    master['meeting_date'] = pd.to_datetime(master['meeting_date'])
    print(f"master_meetings={len(master)}")
    
    # Load LM sentiment
    lm = pd.read_csv(LM_SENTIMENT)
    lm['meeting_date'] = pd.to_datetime(lm['meeting_date'])
    lm_with_data = lm['lm_net_sentiment'].notna().sum()
    print(f"lm_meetings_with_data={lm_with_data}/{len(lm)}")
    
    # Load FinBERT sentiment
    fb = pd.read_csv(FINBERT_SENTIMENT)
    fb['meeting_date'] = pd.to_datetime(fb['meeting_date'])
    fb_with_data = fb['finbert_net_sentiment'].notna().sum()
    print(f"finbert_meetings_with_data={fb_with_data}/{len(fb)}")
    
    return master, lm, fb

def merge_datasets(master, lm, fb):
    """Merge all datasets on meeting_date."""
    # Start with master dataset
    merged = master.merge(lm, on='meeting_date', how='left', suffixes=('', '_lm')) # how='left' so we keep all meetings in master file even if we had NA for sentiment.
    print(f"rows_after_lm_merge={len(merged)}")
    
    # Add FinBERT sentiment after merging LM, just for readability. Could merge all at once, but I prefer this for clarity.
    merged = merged.merge(fb, on='meeting_date', how='left', suffixes=('', '_fb'))
    print(f"rows_after_finbert_merge={len(merged)}")
    
    # Check for duplicate decision columns and clean up. Bit of a safety check as the sentiment CSVs already have a decision_y column (policy decision).
    # To avoid confusion and potential issues with duplication, best to drop any redundant columns
    if 'decision_y_lm' in merged.columns:
        merged.drop(columns=['decision_y_lm'], inplace=True)
    if 'decision_y_fb' in merged.columns:
        merged.drop(columns=['decision_y_fb'], inplace=True)
    
    return merged

# So obviously this is a vital component in the analysis, as LM and FinBERT are on completely different scales. In order to make the coefficients
# comparable at all, standardization is necessary. Intention is to give mean=0 and std=1 for both, so that a 1 unit change in the variables corresponds
# to a 1 standard deviation change in sentiment, for both, and as such is directly comparable. The docstring explains the Z-scoring logic.
def add_z_scores(df):
    """
    Add Z-score standardized sentiment variables.
    
    Z-score formula: (x - mean) / std
    Result: mean=0, std=1
    
    Interpretation in regression:
    Coefficient = effect of 1 standard deviation change in sentiment
    
    This makes LM and FinBERT coefficients directly comparable!
    """
    # LM standardization
    lm_mean = df['lm_net_sentiment'].mean()
    lm_std = df['lm_net_sentiment'].std()
    print(f"lm_mean={lm_mean:+.4f}, lm_std={lm_std:.4f}")
    
    df['lm_sentiment_std'] = (df['lm_net_sentiment'] - lm_mean) / lm_std
    
    # Verify standardization worked
    new_mean = df['lm_sentiment_std'].mean()
    new_std = df['lm_sentiment_std'].std()
    print(f"lm_z_mean={new_mean:+.4f}, lm_z_std={new_std:.4f}")
    assert abs(new_mean) < 1e-10, "Z-score mean should be ~0" # Trying to get mean and std as close as possible to 0
    assert abs(new_std - 1.0) < 1e-10, "Z-score std should be ~1" # Same here, but 1 of course. Allow for some very, very minor imprecision.
    
    # FinBERT standardization
    fb_mean = df['finbert_net_sentiment'].mean()
    fb_std = df['finbert_net_sentiment'].std()
    print(f"finbert_mean={fb_mean:+.4f}, finbert_std={fb_std:.4f}")
    
    df['finbert_sentiment_std'] = (df['finbert_net_sentiment'] - fb_mean) / fb_std
    
    # Verify
    new_mean = df['finbert_sentiment_std'].mean()
    new_std = df['finbert_sentiment_std'].std()
    print(f"finbert_z_mean={new_mean:+.4f}, finbert_z_std={new_std:.4f}")
    assert abs(new_mean) < 1e-10, "Z-score mean should be ~0" # Same as the above for FinBERT mean and std
    assert abs(new_std - 1.0) < 1e-10, "Z-score std should be ~1"
    
    return df

# So for this function I believe the docstring illustrates it quite well (just making som comments in retrospect to clear everything up)
# The use of Int64 is to allow for NA values, but really just an artifact from a previous iteration of the code where I had some meetings with missing decisions.
# Int64 apparently allows for NaN without crashing, which is not necessarily the case for .astype(int).
# Anyway, those issues are obviously resolved now, but keeping Int64 is robust anyway
def create_decision_ordinal(df):
    """
    Create decision_ordinal: ordinal encoding of the policy decision.

    decision_y contains the actual rate change in percentage points
    (e.g. -0.50, -0.25, 0.00, +0.25, +0.50).
    decision_ordinal collapses this to a three-category ordinal variable:
        -1  : any rate cut  (decision_y < 0)
         0  : no change     (decision_y == 0)
        +1  : any rate hike (decision_y > 0)

    This is the dependent variable used in all Ordered Probit regressions.
    """
    import numpy as np

# np.sign() returns -1, 0 or +1 depending on the sign of the input, which is very useful for this exact purpose,
# where we are differentiating between cuts, holds and hikes.
    df['decision_ordinal'] = np.sign(df['decision_y']).astype('Int64')

    cuts  = (df['decision_ordinal'] == -1).sum()
    holds = (df['decision_ordinal'] ==  0).sum()
    hikes = (df['decision_ordinal'] ==  1).sum()

    print(f"\ndecision_distribution_n={len(df)}")
    print(f"cuts={cuts} ({cuts/len(df)*100:.1f}%)")
    print(f"holds={holds} ({holds/len(df)*100:.1f}%)")
    print(f"hikes={hikes} ({hikes/len(df)*100:.1f}%)")

    return df

# This function doesnt really mean anything for the analysis, its just for descriptive purposes to visualize the
# actual change in basis points... 
def create_rate_change_variable(df):
    """
    Create rate_change_bps: the actual rate change in basis points.

    decision_y is already the rate change in percentage points, so
    multiply by 100 to express in basis points.
    This column is retained for descriptive analysis only, the
    Ordered Probit uses decision_ordinal.
    """
    df['rate_change_bps'] = (df['decision_y'] * 100).round(0).astype('Int64')

    print("\nrate_change_bps_distribution:")
    print(df['rate_change_bps'].value_counts().sort_index().to_string())

    return df

# Just for overview of the dataset prior to saving. Produces no output used in analysis, was just a nice to have when building
def summary_statistics(df):
    """Print summary statistics of final dataset."""
    print(f"\ntotal_meetings={len(df)}")
    print(f"date_range={df['meeting_date'].min().date()} to {df['meeting_date'].max().date()}")
    
    lm_available = df['lm_net_sentiment'].notna().sum()
    fb_available = df['finbert_net_sentiment'].notna().sum()
    both_available = (df['lm_net_sentiment'].notna() & df['finbert_net_sentiment'].notna()).sum()
    
    print(f"lm_available={lm_available}/{len(df)} ({lm_available/len(df)*100:.1f}%)")
    print(f"finbert_available={fb_available}/{len(df)} ({fb_available/len(df)*100:.1f}%)")
    print(f"both_available={both_available}/{len(df)} ({both_available/len(df)*100:.1f}%)")
    
    print("\nkey_variables:")
    key_vars = ['lm_net_sentiment', 'finbert_net_sentiment', 'lm_sentiment_std', 
                'finbert_sentiment_std', 'inflation_gap', 'output_next', 'lagged_rate']
    
    for var in key_vars:
        if var in df.columns:
            n = df[var].notna().sum()
            mean = df[var].mean()
            std = df[var].std()
            print(f"  {var}: n={n}, mean={mean:+.4f}, std={std:.4f}")
    
    # Correlation between LM and FinBERT
    complete = df[['lm_net_sentiment', 'finbert_net_sentiment']].dropna()
    if len(complete) > 0:
        corr = complete['lm_net_sentiment'].corr(complete['finbert_net_sentiment'])
        print(f"lm_finbert_corr={corr:.3f} (n={len(complete)})")

def save_output(df):
    """Save final merged dataset."""
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\nsaved: {OUTPUT_FILE}")
    print(f"rows={len(df)}")
    print(f"columns={len(df.columns)}")
    print(f"file_size_kb={Path(OUTPUT_FILE).stat().st_size / 1024:.1f}")

# main() runs everything in the desired order
def main():
    """Main pipeline."""
    # Load data
    master, lm, fb = load_data()
    
    # Merge
    merged = merge_datasets(master, lm, fb)

    # Add Z-scores
    merged = add_z_scores(merged)

    # Add decision_ordinal (-1/0/+1) - dependent variable for Ordered Probit
    merged = create_decision_ordinal(merged)

    # Add rate_change_bps - for descriptive analysis only
    merged = create_rate_change_variable(merged)
    
    # Summary statistics
    summary_statistics(merged)
    
    # Save
    save_output(merged)

if __name__ == "__main__":
    main()
