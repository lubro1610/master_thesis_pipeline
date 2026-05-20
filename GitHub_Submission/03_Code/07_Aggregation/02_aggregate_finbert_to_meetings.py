"""Aggregate FinBERT sentiment to meeting-window level."""

import pandas as pd
import os
from datetime import datetime

INPUT_DIR = "02_Data/gold_corpus_finbert"
OUTPUT_DIR = "02_Data/sentiment_meeting_level"
MASTER_DATA = "02_Data/master_dataset_2006_2025.csv"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "finbert_sentiment_meetings.csv")

INPUT_FILES = [
    "gold_finbert_speeches.csv",
    "gold_finbert_press_releases.csv",
    "gold_finbert_mpr.csv",
    "gold_finbert_finstab.csv",
    "gold_finbert_banklend.csv"
]

def load_all_finbert_data():
    """Load and concatenate all FinBERT scored files."""

    dfs = []
    for file in INPUT_FILES:
        filepath = os.path.join(INPUT_DIR, file)
        if not os.path.exists(filepath):
            print(f"  WARNING: {file} not found, skipping")
            continue
        
        df = pd.read_csv(filepath)
        dfs.append(df)
        print(f"{file}: {len(df):,} sentences")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"total_sentences={len(combined):,}")
    return combined

def aggregate_to_document_level(df):
    """Aggregate sentence-level scores to document-level averages."""
    
    # Group by actual document identifiers (date + title + category), NOT doc_id!
    # doc_id is just a sequential index within each file and NOT unique across files
    doc_level = df.groupby(['date', 'title', 'category']).agg({
        'finbert_net_sentiment': 'mean',
        'finbert_positive_score': 'mean',
        'finbert_negative_score': 'mean',
        'finbert_neutral_score': 'mean',
        'sentence': 'count'
    }).reset_index()
    
    doc_level.rename(columns={'sentence': 'num_sentences'}, inplace=True)
    
    print(f"documents={len(doc_level):,}")
    print(f"avg_sentences_per_document={doc_level['num_sentences'].mean():.1f}")
    print(f"mean_document_net_sentiment={doc_level['finbert_net_sentiment'].mean():+.4f}")
    
    return doc_level

def load_meeting_dates():
    """Load meeting dates from master dataset."""

    master = pd.read_csv(MASTER_DATA)
    master['meeting_date'] = pd.to_datetime(master['meeting_date'])
    meetings = master[['meeting_date', 'decision_y']].copy()
    meetings = meetings.sort_values('meeting_date').reset_index(drop=True)
    
    print(f"meetings={len(meetings)}, {meetings['meeting_date'].min().date()} to {meetings['meeting_date'].max().date()}")
    
    return meetings

def assign_documents_to_windows(doc_df, meetings):
    """Assign each document to the appropriate inter-meeting window."""
    
    # Convert dates
    doc_df['date'] = pd.to_datetime(doc_df['date'])
    
    meeting_assignments = []
    
    for i in range(len(meetings)):
        curr_date = meetings.loc[i, 'meeting_date']
        
        # Define window start
        if i == 0:
            # Start from the previous Norges Bank meeting outside the sample.
            prev_date = pd.Timestamp('2005-12-14')
        else:
            # Subsequent meetings: start from previous meeting date
            prev_date = meetings.loc[i-1, 'meeting_date']
        
        # Include documents on the previous meeting date, exclude the current meeting date.
        mask = (doc_df['date'] >= prev_date) & (doc_df['date'] < curr_date)
        docs_in_window = doc_df[mask].copy()
        
        if len(docs_in_window) > 0:
            docs_in_window['meeting_date'] = curr_date
            meeting_assignments.append(docs_in_window)
    
    assigned_docs = pd.concat(meeting_assignments, ignore_index=True) if meeting_assignments else pd.DataFrame()
    
    print(f"documents_assigned={len(assigned_docs):,}")
    print(f"documents_unassigned={len(doc_df) - len(assigned_docs):,}")
    print(f"avg_documents_per_meeting={len(assigned_docs) / len(meetings):.1f}")
    
    return assigned_docs

def aggregate_to_meeting_level(assigned_docs, meetings):
    """Aggregate document-level scores to meeting-level averages."""
    
    # Average across documents for each meeting
    # Count documents by counting unique (date, title, category) combinations
    meeting_sentiment = assigned_docs.groupby('meeting_date').agg({
        'finbert_net_sentiment': 'mean',
        'finbert_positive_score': 'mean',
        'finbert_negative_score': 'mean',
        'finbert_neutral_score': 'mean',
        'title': 'count',
        'num_sentences': 'sum'
    }).reset_index()
    
    meeting_sentiment.rename(columns={
        'title': 'num_documents',
        'num_sentences': 'num_sentences_total'
    }, inplace=True)
    
    # Merge with full meeting list (some meetings may have no documents)
    result = meetings.merge(
        meeting_sentiment, 
        on='meeting_date', 
        how='left'
    )
    
    meetings_with_data = result['num_documents'].notna().sum()
    
    print(f"meetings_with_data={meetings_with_data}/{len(result)}")
    print(f"mean_meeting_net_sentiment={result['finbert_net_sentiment'].mean():+.4f}")
    print(f"std_meeting_net_sentiment={result['finbert_net_sentiment'].std():.4f}")
    
    return result

def save_output(df):
    """Save meeting-level FinBERT sentiment to CSV."""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"saved: {OUTPUT_FILE}")
    print(f"rows={len(df)}")
    print(f"columns={len(df.columns)}")

def main():
    """Main aggregation pipeline."""

    sentence_df = load_all_finbert_data()
    
    doc_df = aggregate_to_document_level(sentence_df)
    
    meetings = load_meeting_dates()
    
    assigned_docs = assign_documents_to_windows(doc_df, meetings)
    
    meeting_df = aggregate_to_meeting_level(assigned_docs, meetings)
    
    save_output(meeting_df)

if __name__ == "__main__":
    main()
