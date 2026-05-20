# Scores each sentence in the gold corpus using the Loughran-McDonald dictionary.
# LM and FinBERT both get the same raw text - that was a deliberate choice so any
# difference in results reflects the methods themselves, not data availability.
# An earlier version stripped stopwords for LM, which wasn't really fair to FinBERT.
#
# Output is sentence-level scores (counts + ratios per category).
# Net sentiment (positive - negative ratio) is what ends up in the regression.
# Topic mention counts are saved but not used - kept them around in case I wanted
# to try an alternative specification later.
#
# Input:  02_Data/gold_corpus/*.csv
# Output: 02_Data/gold_corpus_lm/*.csv

import pandas as pd
import re
from pathlib import Path
from typing import Set, Dict

# Notice here:Topic mention keywords - saved to file but not used in regression.
# If youre reading this, you might be wondering why I have these topic keywords and counts
# in the code since they dont end up in regression. I originally thought I might want to try
# something different, but ended up with the net sentiment measure because it was less complex
# and more interpretable and comparable to FinBERT. I kept the code in case it would be useful later
# on, to kind of illustrate the creative process, and now I leave it because I dont want to 
# cause any bugs by deleting it since it goes into the master dataset.
TOPIC_KEYWORDS = {
    'inflation': {
        'inflation', 'deflation', 'cpi', 'consumer_price_index', 'price_level',
        'price_stability', 'price_pressure', 'price_growth', 'cost_growth',
        'wage_growth', 'price_index'
    },
    'unemployment': {
        'unemployment', 'unemployed', 'jobless', 'employment', 'jobs', 'hiring',
        'labor_market', 'labour_market', 'participation_rate', 'labor_force',
        'labour_force', 'slack'
    },
    'recession': {
        'recession', 'economic_contraction', 'downturn', 'slowdown', 'decline',
        'negative_growth', 'contraction', 'crisis', 'depression', 'output_fall'
    },
    'lending': {
        'lending', 'borrowing', 'credit', 'loan', 'loans', 'mortgage', 'mortgages',
        'credit_conditions', 'credit_availability', 'credit_spread', 'bank_lending'
    }
}

def load_lm_dictionary(dict_path: Path) -> Dict[str, Set[str]]:
    # Reads lm_dict.csv and builds one set of words per category.
    # A word belongs to a category if its column value is > 0.
    df = pd.read_csv(dict_path)
    
    categories = ['Negative', 'Positive', 'Uncertainty', 'Litigious', 
                  'Strong_Modal', 'Weak_Modal', 'Constraining']
    
    word_categories = {cat: set() for cat in categories}
    
    for _, row in df.iterrows():
        word = str(row['Word']).strip().lower()
        
        for cat in categories:
            if pd.notna(row[cat]) and row[cat] > 0:
                word_categories[cat].add(word)
    
    print("lm_dictionary_words:")
    for cat in categories:
        print(f"  {cat}: {len(word_categories[cat]):,} words")
    
    return word_categories

def tokenize_and_clean(text: str) -> list:
    # Lowercase and pull out alphabetic tokens only - numbers and punctuation dropped.
    # Stopwords kept so the denominator matches what FinBERT sees.
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    return tokens


def calculate_lm_scores(tokens: list, word_categories: Dict[str, Set[str]]) -> Dict:
    # Counts how many tokens fall in each LM category, then divides by total tokens.
    # max(..., 1) just avoids a division by zero on empty sentences.
    categories = ['Negative', 'Positive', 'Uncertainty', 'Litigious', 
                  'Strong_Modal', 'Weak_Modal', 'Constraining']
    
    total_tokens = len(tokens)
    result = {'lm_total_tokens': total_tokens}
    
    for cat in categories:
        cat_lower = cat.lower()
        count = sum(1 for token in tokens if token in word_categories[cat])
        ratio = count / max(total_tokens, 1)
        
        result[f'lm_{cat_lower}_count'] = count
        result[f'lm_{cat_lower}_ratio'] = ratio
    
    return result


def count_topic_mentions(tokens: list) -> Dict:
    # Counts how many times each topic appears in the sentence.
    # Not used in the regression, just saved in case it becomes useful.
    result = {}
    
    for topic, keywords in TOPIC_KEYWORDS.items():
        mentions = sum(1 for token in tokens if token in keywords)
        result[f'topic_{topic}_mentions'] = mentions
    
    return result


def process_corpus_file(input_path: Path, output_path: Path, 
                       word_categories: Dict[str, Set[str]]) -> None:
    # Scores every sentence in one corpus file and saves the result.
    print(f"\n{input_path.name}")
    
    df = pd.read_csv(input_path)
    print(f"sentences={len(df):,}")
    
    categories = ['Negative', 'Positive', 'Uncertainty', 'Litigious', 
                  'Strong_Modal', 'Weak_Modal', 'Constraining']
    
    # Pre-fill columns with zeros so every row has a value even if the sentence is empty
    for cat in categories:
        df[f'lm_{cat.lower()}_count'] = 0
        df[f'lm_{cat.lower()}_ratio'] = 0.0
    df['lm_total_tokens'] = 0
    for topic in TOPIC_KEYWORDS.keys():
        df[f'topic_{topic}_mentions'] = 0
    
    for idx, row in df.iterrows():
        tokens = tokenize_and_clean(row['sentence'])
        lm_scores = calculate_lm_scores(tokens, word_categories)
        for key, value in lm_scores.items():
            if key in df.columns:
                df.at[idx, key] = value
        topic_mentions = count_topic_mentions(tokens)
        for key, value in topic_mentions.items():
            if key in df.columns:
                df.at[idx, key] = value
    
    avg_net = (df['lm_positive_ratio'] - df['lm_negative_ratio']).mean()
    print(f"avg_negative_ratio={df['lm_negative_ratio'].mean():.4f}")
    print(f"avg_positive_ratio={df['lm_positive_ratio'].mean():.4f}")
    print(f"avg_net_sentiment={avg_net:.4f}")
    
    df.to_csv(output_path, index=False)
    print(f"saved: {output_path.name}")

def main():
    base_dir = Path(__file__).parent.parent.parent
    dict_path = base_dir / "Environment" / "lm_dict.csv"
    gold_corpus_dir = base_dir / "02_Data" / "gold_corpus"
    output_dir = base_dir / "02_Data" / "gold_corpus_lm"
    
    if not dict_path.exists():
        print(f"\nERROR: Dictionary not found at {dict_path}")
        return
    
    word_categories = load_lm_dictionary(dict_path)
    output_dir.mkdir(exist_ok=True)
    
    files = [
        'gold_speeches.csv',
        'gold_press_releases.csv',
        'gold_mpr.csv',
        'gold_finstab.csv',
        'gold_banklend.csv'
    ]
    
    for filename in files:
        input_path = gold_corpus_dir / filename
        output_path = output_dir / filename.replace('gold_', 'gold_lm_')
        
        if not input_path.exists():
            print(f"\nWarning: {filename} not found, skipping...")
            continue
        
        process_corpus_file(input_path, output_path, word_categories)
    
    print(f"\noutput_dir={output_dir}")


if __name__ == "__main__":
    main()
