"""Aggregate LM sentiment to meeting-window level."""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

LM_SCORE_COLUMNS = [
    'lm_negative_ratio', 'lm_positive_ratio', 'lm_uncertainty_ratio',
    'lm_litigious_ratio', 'lm_strong_modal_ratio', 'lm_weak_modal_ratio',
    'lm_constraining_ratio'
]

LM_COUNT_COLUMNS = [
    'lm_negative_count', 'lm_positive_count', 'lm_uncertainty_count',
    'lm_litigious_count', 'lm_strong_modal_count', 'lm_weak_modal_count',
    'lm_constraining_count'
]

TOPIC_COLUMNS = [
    'topic_inflation_mentions', 'topic_unemployment_mentions',
    'topic_recession_mentions', 'topic_lending_mentions'
]

# No fixed window: use between-meetings logic instead.

def load_all_lm_sentence_data(lm_dir: Path) -> pd.DataFrame:
    """Load all sentence-level LM data."""
    
    files = [
        'gold_lm_speeches.csv',
        'gold_lm_press_releases.csv',
        'gold_lm_mpr.csv',
        'gold_lm_finstab.csv',
        'gold_lm_banklend.csv'
    ]
    
    dfs = []
    for filename in files:
        filepath = lm_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            df['source_file'] = filename.replace('gold_lm_', '').replace('.csv', '')
            dfs.append(df)
            print(f"{filename}: {len(df):,} sentences")
        else:
            print(f"warning: {filename} not found")
    
    all_data = pd.concat(dfs, ignore_index=True)
    
    all_data['date'] = pd.to_datetime(all_data['date'], errors='coerce')
    
    print(f"total_sentences={len(all_data):,}")
    
    return all_data


def aggregate_to_document_level(sentence_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate sentence-level scores to documents using pool-then-ratio."""
    
    grouping_cols = ['date', 'title', 'category']
    
    agg_dict = {}
    
    # Sum of count columns (pool all words per document)
    for col in LM_COUNT_COLUMNS:
        if col in sentence_df.columns:
            agg_dict[col] = 'sum'
    
    # Sum of topic mentions
    for col in TOPIC_COLUMNS:
        if col in sentence_df.columns:
            agg_dict[col] = 'sum'
    
    # Count sentences per document
    agg_dict['sentence'] = 'count'
    
    # Sum tokens per document
    if 'lm_total_tokens' in sentence_df.columns:
        agg_dict['lm_total_tokens'] = 'sum'
    
    doc_df = sentence_df.groupby(grouping_cols, as_index=False).agg(agg_dict)
    
    # Recalculate ratios from pooled counts (Pool-then-ratio method)
    # This gives each word equal weight, not each sentence
    categories = ['negative', 'positive', 'uncertainty', 'litigious',
                  'strong_modal', 'weak_modal', 'constraining']
    
    for cat in categories:
        count_col = f'lm_{cat}_count'
        ratio_col = f'lm_{cat}_ratio'
        if count_col in doc_df.columns and 'lm_total_tokens' in doc_df.columns:
            doc_df[ratio_col] = doc_df[count_col] / doc_df['lm_total_tokens'].replace(0, 1)
    
    doc_df = doc_df.rename(columns={'sentence': 'sentence_count'})
    
    print(f"documents={len(doc_df):,}")
    print(f"avg_sentences_per_document={doc_df['sentence_count'].mean():.1f}")
    print(f"avg_tokens_per_document={doc_df['lm_total_tokens'].mean():.0f}")
    
    return doc_df


def load_meeting_dates(master_path: Path) -> pd.DataFrame:
    """Load meeting dates from the master dataset."""
    
    df = pd.read_csv(master_path)
    df['meeting_date'] = pd.to_datetime(df['meeting_date'])
    
    print(f"meetings={len(df)}, {df['meeting_date'].min().date()} to {df['meeting_date'].max().date()}")
    
    return df[['meeting_date']].copy()


def aggregate_to_meeting_level(doc_df: pd.DataFrame, 
                               meetings_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate document-level scores to meeting-window level."""
    
    # Ensure meetings are sorted
    meetings_df = meetings_df.sort_values('meeting_date').reset_index(drop=True)
    
    results = []
    
    for idx, meeting_row in meetings_df.iterrows():
        meeting_date = meeting_row['meeting_date']
        
        # Define window start: previous meeting date
        if idx == 0:
            # Start from the previous Norges Bank meeting outside the sample.
            window_start = pd.Timestamp('2005-12-14')
        else:
            # Subsequent meetings: start from previous meeting date (exclusive)
            window_start = meetings_df.loc[idx - 1, 'meeting_date']
        
        # Window end: current meeting date (exclusive - documents must be published BEFORE meeting)
        window_end = meeting_date
        
        # Filter documents in window
        # Use >= for window_start to include documents published ON previous meeting date
        # (they represent communication since last decision)
        # Use < for window_end to exclude documents from current meeting date (avoid look-ahead)
        docs_in_window = doc_df[
            (doc_df['date'] >= window_start) & 
            (doc_df['date'] < window_end)
        ].copy()
        
        n_docs = len(docs_in_window)
        
        if n_docs == 0:
            # No documents in window - all scores NaN
            result = {'meeting_date': meeting_date, 'n_documents': 0, 'n_sentences': 0}
            for col in LM_SCORE_COLUMNS + LM_COUNT_COLUMNS + TOPIC_COLUMNS:
                result[col] = np.nan
        else:
            # Aggregate: mean of document ratios (Two-Step Pool approach)
            # Each document counts equally, preventing long documents from dominating
            result = {'meeting_date': meeting_date, 'n_documents': n_docs}
            
            # Mean of ratios (each document = 1 vote)
            for col in LM_SCORE_COLUMNS:
                if col in docs_in_window.columns:
                    result[col] = docs_in_window[col].mean()
            
            # Sum of counts
            for col in LM_COUNT_COLUMNS:
                if col in docs_in_window.columns:
                    result[col] = docs_in_window[col].sum()
            
            # Sum of topic mentions
            for col in TOPIC_COLUMNS:
                if col in docs_in_window.columns:
                    result[col] = docs_in_window[col].sum()
            
            # Total sentences
            if 'sentence_count' in docs_in_window.columns:
                result['n_sentences'] = docs_in_window['sentence_count'].sum()
            
            # Total tokens
            if 'lm_total_tokens' in docs_in_window.columns:
                result['total_tokens'] = docs_in_window['lm_total_tokens'].sum()
        
        results.append(result)
    
    meeting_sentiment = pd.DataFrame(results)
    
    meetings_with_data = (meeting_sentiment['n_documents'] > 0).sum()
    print(f"meeting_rows={len(meeting_sentiment)}")
    print(f"meetings_with_data={meetings_with_data}/{len(meeting_sentiment)} ({meetings_with_data/len(meeting_sentiment)*100:.1f}%)")
    print(f"avg_documents_per_meeting={meeting_sentiment['n_documents'].mean():.1f}")
    print(f"avg_sentences_per_meeting={meeting_sentiment['n_sentences'].mean():.0f}")
    
    return meeting_sentiment


def add_net_sentiment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Add net sentiment scores as positive minus negative."""
    
    if 'lm_positive_ratio' in df.columns and 'lm_negative_ratio' in df.columns:
        df['lm_net_sentiment'] = df['lm_positive_ratio'] - df['lm_negative_ratio']
        
        print(f"lm_net_sentiment_min={df['lm_net_sentiment'].min():.4f}")
        print(f"lm_net_sentiment_max={df['lm_net_sentiment'].max():.4f}")
        print(f"lm_net_sentiment_mean={df['lm_net_sentiment'].mean():.4f}")
    
    return df


def main():
    """Main execution pipeline."""

    base_dir = Path(__file__).parent.parent.parent
    lm_dir = base_dir / "02_Data" / "gold_corpus_lm"
    master_path = base_dir / "02_Data" / "master_dataset_2006_2025.csv"
    output_dir = base_dir / "02_Data" / "sentiment_meeting_level"
    
    output_dir.mkdir(exist_ok=True)
    
    sentence_df = load_all_lm_sentence_data(lm_dir)
    
    doc_df = aggregate_to_document_level(sentence_df)
    
    meetings_df = load_meeting_dates(master_path)
    meeting_sentiment = aggregate_to_meeting_level(doc_df, meetings_df)
    
    meeting_sentiment = add_net_sentiment_scores(meeting_sentiment)
    
    output_path = output_dir / "lm_sentiment_meetings.csv"
    meeting_sentiment.to_csv(output_path, index=False)
    
    print(f"saved: {output_path}")
    print(f"rows={len(meeting_sentiment)}")
    print(f"columns={len(meeting_sentiment.columns)}")
    
if __name__ == "__main__":
    main()
