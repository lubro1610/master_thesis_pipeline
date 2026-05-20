"""Document-type heterogeneity analysis for sentiment coefficients."""

import sys
import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
from scipy import stats
from scipy.stats import norm, spearmanr

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import warnings
warnings.filterwarnings('ignore')

finbert_files = {
    'speeches': '02_Data/gold_corpus_finbert/gold_finbert_speeches.csv',
    'press_releases': '02_Data/gold_corpus_finbert/gold_finbert_press_releases.csv',
    'mpr': '02_Data/gold_corpus_finbert/gold_finbert_mpr.csv',
    'finstab': '02_Data/gold_corpus_finbert/gold_finbert_finstab.csv',
    'banklend': '02_Data/gold_corpus_finbert/gold_finbert_banklend.csv'
}

lm_files = {
    'speeches': '02_Data/gold_corpus_lm/gold_lm_speeches.csv',
    'press_releases': '02_Data/gold_corpus_lm/gold_lm_press_releases.csv',
    'mpr': '02_Data/gold_corpus_lm/gold_lm_mpr.csv',
    'finstab': '02_Data/gold_corpus_lm/gold_lm_finstab.csv',
    'banklend': '02_Data/gold_corpus_lm/gold_lm_banklend.csv'
}

finbert_dfs = {}
for doc_type, filepath in finbert_files.items():
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['doc_type'] = doc_type
    finbert_dfs[doc_type] = df
    print(f"FinBERT {doc_type}: {len(df):,} sentences")

finbert_all = pd.concat(finbert_dfs.values(), ignore_index=True)

lm_dfs = {}
for doc_type, filepath in lm_files.items():
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df['doc_type'] = doc_type
    lm_dfs[doc_type] = df
    print(f"LM {doc_type}: {len(df):,} sentences")

lm_all = pd.concat(lm_dfs.values(), ignore_index=True)

print(f"total FinBERT sentences: {len(finbert_all):,}")
print(f"total LM sentences: {len(lm_all):,}")

def aggregate_finbert_to_docs(df):
    """Aggregate FinBERT sentence scores to document means."""
    doc_agg = df.groupby(['date', 'title', 'doc_type']).agg({
        'finbert_net_sentiment': 'mean',
        'sentence': 'count'
    }).reset_index()
    doc_agg.columns = ['date', 'title', 'doc_type', 'finbert_sentiment', 'n_sentences']
    return doc_agg

def aggregate_lm_to_docs(df):
    """Aggregate LM counts before computing document-level ratios."""
    doc_agg = df.groupby(['date', 'title', 'doc_type']).agg({
        'lm_positive_count': 'sum',
        'lm_negative_count': 'sum',
        'lm_total_tokens': 'sum',
        'sentence': 'count'
    }).reset_index()
    
    doc_agg['lm_sentiment'] = (doc_agg['lm_positive_count'] - doc_agg['lm_negative_count']) / doc_agg['lm_total_tokens']
    doc_agg = doc_agg.rename(columns={'sentence': 'n_sentences'})
    
    return doc_agg[['date', 'title', 'doc_type', 'lm_sentiment', 'n_sentences']]

finbert_docs = aggregate_finbert_to_docs(finbert_all)
lm_docs = aggregate_lm_to_docs(lm_all)

print(f"FinBERT documents: {len(finbert_docs):,}")
print(f"LM documents: {len(lm_docs):,}")

print("\nDocuments per type:")
print(finbert_docs['doc_type'].value_counts().sort_index())

meetings = pd.read_csv('02_Data/master_dataset_2006_2025.csv')
meetings['meeting_date'] = pd.to_datetime(meetings['meeting_date'])
meetings = meetings.sort_values('meeting_date').reset_index(drop=True)

print(f"Meetings: {len(meetings)}")

def assign_to_meeting_windows(docs_df, meetings_df):
    """Assign documents to inter-meeting windows without look-ahead."""
    docs_df = docs_df.copy()
    docs_df['meeting_window'] = pd.NaT
    
    for i in range(len(meetings_df)):
        curr_meeting = meetings_df.loc[i, 'meeting_date']
        
        if i == 0:
            window_start = pd.Timestamp('2005-12-14')
        else:
            window_start = meetings_df.loc[i-1, 'meeting_date']
        
        window_end = curr_meeting
        
        mask = (docs_df['date'] >= window_start) & (docs_df['date'] < window_end)
        docs_df.loc[mask, 'meeting_window'] = curr_meeting
    
    docs_assigned = docs_df[docs_df['meeting_window'].notna()].copy()
    
    return docs_assigned

finbert_assigned = assign_to_meeting_windows(finbert_docs, meetings)
lm_assigned = assign_to_meeting_windows(lm_docs, meetings)

print(f"FinBERT docs assigned: {len(finbert_assigned):,}")
print(f"LM docs assigned: {len(lm_assigned):,}")

def aggregate_by_doctype_and_meeting(docs_df, sentiment_col):
    """Aggregate document means separately by document type and meeting."""
    meeting_agg = docs_df.groupby(['meeting_window', 'doc_type']).agg({
        sentiment_col: 'mean',
        'title': 'count'
    }).reset_index()
    
    meeting_agg.columns = ['meeting_window', 'doc_type', sentiment_col, 'n_docs']
    
    meeting_wide = meeting_agg.pivot(
        index='meeting_window',
        columns='doc_type',
        values=sentiment_col
    ).reset_index()
    
    meeting_wide.columns = ['meeting_date'] + [f'{col}_{sentiment_col}' for col in meeting_wide.columns[1:]]
    
    return meeting_wide

finbert_by_type = aggregate_by_doctype_and_meeting(finbert_assigned, 'finbert_sentiment')
lm_by_type = aggregate_by_doctype_and_meeting(lm_assigned, 'lm_sentiment')

print(f"Meetings with FinBERT document-type data: {len(finbert_by_type)}")
print(f"Meetings with LM document-type data: {len(lm_by_type)}")

print("\nDocument type coverage (% meetings with ≥1 doc):")
for doc_type in ['speeches', 'press_releases', 'mpr', 'finstab', 'banklend']:
    col_name = f'{doc_type}_finbert_sentiment'
    if col_name in finbert_by_type.columns:
        coverage = (finbert_by_type[col_name].notna().sum() / len(finbert_by_type) * 100)
        print(f"  {doc_type:20s}: {coverage:.1f}%")

master = pd.read_csv('02_Data/master_dataset_with_sentiment.csv')
master['meeting_date'] = pd.to_datetime(master['meeting_date'])

master_enhanced = master.merge(finbert_by_type, on='meeting_date', how='left')
master_enhanced = master_enhanced.merge(lm_by_type, on='meeting_date', how='left')

doc_types = ['speeches', 'press_releases', 'mpr', 'finstab', 'banklend']

for doc_type in doc_types:
    fb_col = f'{doc_type}_finbert_sentiment'
    lm_col = f'{doc_type}_lm_sentiment'
    
    if fb_col in master_enhanced.columns:
        master_enhanced[f'{doc_type}_finbert_std'] = (
            master_enhanced[fb_col] - master_enhanced[fb_col].mean()
        ) / master_enhanced[fb_col].std()
    
    if lm_col in master_enhanced.columns:
        master_enhanced[f'{doc_type}_lm_std'] = (
            master_enhanced[lm_col] - master_enhanced[lm_col].mean()
        ) / master_enhanced[lm_col].std()

print("merged and standardized document-type sentiment")

print("\ncorrelation with decisions by document type:")

print(f"\n{'Document Type':<20} {'N':<6} {'FinBERT r':<12} {'p-val':<10} {'LM r':<12} {'p-val':<10}")

for doc_type in doc_types:
    fb_col = f'{doc_type}_finbert_std'
    lm_col = f'{doc_type}_lm_std'
    
    if fb_col in master_enhanced.columns and lm_col in master_enhanced.columns:
        subset = master_enhanced[[fb_col, lm_col, 'decision_ordinal']].dropna()
        
        if len(subset) > 10:
            fb_corr, fb_pval = spearmanr(subset[fb_col], subset['decision_ordinal'])
            lm_corr, lm_pval = spearmanr(subset[lm_col], subset['decision_ordinal'])
            
            fb_sig = '***' if fb_pval < 0.01 else '**' if fb_pval < 0.05 else '*' if fb_pval < 0.10 else ''
            lm_sig = '***' if lm_pval < 0.01 else '**' if lm_pval < 0.05 else '*' if lm_pval < 0.10 else ''
            
            print(f"{doc_type:<20} {len(subset):<6} {fb_corr:+.3f} {fb_sig:<4}  {fb_pval:<10.4f} {lm_corr:+.3f} {lm_sig:<4}  {lm_pval:<10.4f}")

print("\nordered probit regressions by document type:")

base_controls = ['inflation_gap', 'output_next', 'lagged_rate']

print(f"\n{'Document Type':<20} {'Method':<10} {'Coef':<8} {'p-val':<10} {'N':<6} {'Signif':<10}")

results_data = []

for doc_type in doc_types:
    fb_col = f'{doc_type}_finbert_std'
    lm_col = f'{doc_type}_lm_std'
    
    if fb_col in master_enhanced.columns and lm_col in master_enhanced.columns:
        subset = master_enhanced[base_controls + [fb_col, lm_col, 'decision_ordinal']].dropna()
        
        if len(subset) >= 50:
            y = subset['decision_ordinal']
            
            X_fb = subset[base_controls + [fb_col]]
            try:
                model_fb = OrderedModel(y, X_fb, distr='probit')
                result_fb = model_fb.fit(method='bfgs', disp=False)
                
                fb_coef = result_fb.params[fb_col]
                fb_pval = result_fb.pvalues[fb_col]
                fb_sig = '***' if fb_pval < 0.01 else '**' if fb_pval < 0.05 else '*' if fb_pval < 0.10 else ''
                
                print(f"{doc_type:<20} {'FinBERT':<10} {fb_coef:+.4f}  {fb_pval:<10.4f} {len(subset):<6} {fb_sig:<10}")
                
                results_data.append({
                    'doc_type': doc_type,
                    'method': 'FinBERT',
                    'coef': fb_coef,
                    'pval': fb_pval,
                    'n': len(subset)
                })
            except:
                print(f"{doc_type:<20} {'FinBERT':<10} {'FAILED':<8} {'-':<10} {len(subset):<6} {'-':<10}")
            
            X_lm = subset[base_controls + [lm_col]]
            try:
                model_lm = OrderedModel(y, X_lm, distr='probit')
                result_lm = model_lm.fit(method='bfgs', disp=False)
                
                lm_coef = result_lm.params[lm_col]
                lm_pval = result_lm.pvalues[lm_col]
                lm_sig = '***' if lm_pval < 0.01 else '**' if lm_pval < 0.05 else '*' if lm_pval < 0.10 else ''
                
                print(f"{doc_type:<20} {'LM':<10} {lm_coef:+.4f}  {lm_pval:<10.4f} {len(subset):<6} {lm_sig:<10}")
                
                results_data.append({
                    'doc_type': doc_type,
                    'method': 'LM',
                    'coef': lm_coef,
                    'pval': lm_pval,
                    'n': len(subset)
                })
            except:
                print(f"{doc_type:<20} {'LM':<10} {'FAILED':<8} {'-':<10} {len(subset):<6} {'-':<10}")
            
            print()

results_df = pd.DataFrame(results_data)
results_df.to_csv('04_Output/robustness_document_types.csv', index=False)
print("saved: 04_Output/robustness_document_types.csv")

print("\nsignificant document types at 5%:")

# Find which document types are significant for each method
if len(results_df) > 0:
    fb_results = results_df[results_df['method'] == 'FinBERT']
    lm_results = results_df[results_df['method'] == 'LM']
    
    fb_sig = fb_results[fb_results['pval'] < 0.05]
    lm_sig = lm_results[lm_results['pval'] < 0.05]
    
    print(f"FinBERT:")
    if len(fb_sig) > 0:
        for _, row in fb_sig.iterrows():
            print(f"  {row['doc_type']:20s}: beta={row['coef']:+.4f}, p={row['pval']:.4f}")
    else:
        print("  none")
    
    print(f"LM:")
    if len(lm_sig) > 0:
        for _, row in lm_sig.iterrows():
            print(f"  {row['doc_type']:20s}: beta={row['coef']:+.4f}, p={row['pval']:.4f}")
    else:
        print("  none")
