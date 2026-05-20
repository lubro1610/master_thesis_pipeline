# Takes the preprocessed documents and creates the gold corpus - sentence-level CSVs
# ready for LM and FinBERT scoring. Three steps per document: noise removal,
# sentence tokenization, and relevance filtering by keyword matching.
# Press releases skip the keyword filter entirely since they're already focused.
# Imports are the standard libraries as usual, in addition to nltk for sent_tokenize() to split text into sentences.
# Very useful to handle abbreviations and decimal points etc without faulty splitting.
import pandas as pd
import re
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab', quiet=True)

# If youve been following the codebase, you might be wondering why you see even more regex in this file.
# I agree its not the most efficient or elegant design, but I really thought I was happy with the cleaning 
# for this analysis' purpose in the prior scripts, but seeing as I had some to review and extend
# in greater detail I thought why not... Though, the filters could have been implemented in prior scripts
def clean_technical_noise(text):
    # Removes PDF artifacts, page numbers, headers, footers, URLs and boilerplate
    # greeting phrases that add nothing to sentiment scoring.
    
    # Remove page numbers etc (various formats)
    text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Side\s+\d+\s+av\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bSide\s+\d+\b', '', text, flags=re.IGNORECASE)
    
    # Remove common PDF headers/footers
    text = re.sub(r'NORGES\s+BANK\s*\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Monetary\s+Policy\s+Report.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Financial\s+Stability.*?\n', '', text, flags=re.IGNORECASE)
    
    # Remove standalone date stamps (but not dates in sentences)
    text = re.sub(r'^\s*\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*$', 
                  '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'^\s*\d{1,2}\s+(?:januar|februar|mars|april|mai|juni|juli|august|september|oktober|november|desember)\s+\d{4}\s*$', 
                  '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove URLs
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'www\.[^\s]+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    
    # Remove footnote markers
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\(\d+\)', '', text)
    
    # Remove ceremonial boilerplate phrases. I mean (in hindsight)for this type of analysis you could probably go in far greater
    # depth with the boilerplate than what I have, but this was a suggestion from Claude that I thought wouldnt hurt,
    # but it probably doesnt really provide much filtering value.

    boilerplate_patterns = [
        r'Check against delivery',
        r'Thank you for (?:the )?invitation',
        r'It is a pleasure to be here',
        r'Ladies and gentlemen',
        r'Good morning(?:,)?\s*(?:everyone|ladies and gentlemen)?',
        r'Good afternoon(?:,)?\s*(?:everyone|ladies and gentlemen)?',
        r'Thank you for your attention',
    ]
    
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text

def normalize_whitespace(text):
    # Cleans up irregular spacing and line breaks before passing to sent_tokenize.
    # Joins broken lines where the next line starts lowercase (likely mid-sentence).
    
    # Replace tabs with spaces
    text = text.replace('\t', ' ')
    
    # Replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)
    
    # Fix broken line breaks (join lines that dont end with punctuation)
    # but preserve paragraph breaks. For instance, if a sentence is broken apart and then
    # the next line starts with a lowercase letter, it may be treated as a connected sentence and 
    # be accidentally joined.
    lines = text.split('\n')
    fixed_lines = []
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        # If line doesnt end with punctuation and next line exists, join them
        if i < len(lines) - 1 and line and not line[-1] in '.!?:;':
            if lines[i+1].strip() and lines[i+1].strip()[0].islower():
                fixed_lines.append(line)
                continue
        fixed_lines.append(line)
    
    text = ' '.join(fixed_lines)
    
    # Remove space before punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Ensure space after punctuation (include ÆØÅ just in case so we dont run into any weird issues)
    text = re.sub(r'([.,;:!?])([A-ZÆØÅa-zæøå])', r'\1 \2', text)
    
    return text.strip()

def tokenize_sentences(text):
    # NLTK's sent_tokenize handles abbreviations and decimal points well, which matters a lot here
    # ("The U.S. economy..." should not split on the dots).
    
    # NLTK's sent_tokenize works well for both English and Norwegian (not that it should really matter here, but still good practice)
    sentences = sent_tokenize(text)
    
    # Clean each sentence
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def is_relevant_sentence(sentence, is_press_release=False):
    # Keeps a sentence if it contains at least one of the ~138 keywords below.
    # Press releases bypass this entirely - they're already focused enough.
    
    # Exception: press releases are kept regardless
    if is_press_release:
        return True
    
    # Convert to lowercase for matching
    sentence_lower = sentence.lower()
    
    # 138 keywords across 14 categories, using substring matching to catch stems.
    # Could probably be improved - some words are fairly generic - but it works well
    # enough for this purpose and I was fairly conservative in the selection.
    keywords = [
        # === MANDATE & OBJECTIVES (8) ===
        'inflation', 'price', 'cpi', 'target', 'stability', 
        'mandate', 'objective', 'anchor',
        
        # === MONETARY POLICY (15) ===
        'rate', 'policy', 'interest', 'committee', 'tight', 
        'easing', 'ease', 'neutral', 'decision', 'meeting',
        'accommodat', 'restrictive', 'stance', 'transmission', 'signal',
        
        # === ECONOMIC ACTIVITY (15) ===
        'growth', 'gdp', 'output', 'capacit', 'demand', 'activit', 
        'invest', 'consum', 'recession', 'recover',
        'slowdown', 'pickup', 'momentum', 'cyclical', 'expans',
        
        # === LABOR MARKET (8) ===
        'labo', 'employ', 'unemploy', 'wage', 'slack',
        'hiring', 'layoff', 'productiv',
        
        # === EXTERNAL SECTOR (13) ===
        'krone', 'nok', 'exchange', 'oil', 'petrol', 'export', 'import',
        'global', 'abroad', 'trade', 'foreign', 'competitiv', 'commodi',
        
        # === HOUSING & HOUSEHOLD (7) ===
        'hous', 'debt', 'mortgage', 'household', 'saving', 'leverage', 'property',
        
        # === FINANCIAL STABILITY (12) ===
        'bank', 'credit', 'financial', 'liquid', 'lend', 'borrow', 
        'default', 'capital', 'solven', 'resilien', 'buffer', 'provision',
        
        # === FINANCIAL MARKETS (10) ===
        'market', 'equity', 'bond', 'yield', 'spread', 'premium',
        'volatil', 'trading', 'investor', 'asset',
        
        # === RISK & UNCERTAINTY (12) ===
        'risk', 'uncertain', 'vulnerab', 'stress', 'downside', 'upside',
        'tail', 'confiden', 'sentiment', 'outlook', 'balance', 'scenario',
        
        # === CRISIS & SHOCKS (8) ===
        'crisis', 'shock', 'turmoil', 'disrupt', 'contagi',
        'pandemic', 'tension', 'conflict',
        
        # === FORECASTING (6) ===
        'forecast', 'project', 'expect', 'anticipat', 'estimate', 'baseline',
        
        # === FISCAL (5) ===
        'fiscal', 'government', 'budget', 'spending', 'tax',
        
        # === STRUCTURAL (4) ===
        'structural', 'reform', 'potential', 'trend',
        
        # === ECONOMIC TONE/CONDITIONS (15) ===
        'improve', 'deteriorate', 'strengthen', 'weaken', 'elevated',
        'moderate', 'subdued', 'robust', 'fragile', 'solid',
        'challeng', 'concern', 'positive', 'negative', 'adverse',
    ]
    
    # Check if any keyword appears in sentence (substring match)
    for keyword in keywords:
        if keyword in sentence_lower:
            return True
    
    return False

def process_document(text, title, date, category, doc_id):
    # Runs all three stages on a single document and returns a list of sentence dicts.
    
    # Step 1: Clean technical noise
    text = clean_technical_noise(text)
    
    # Normalize whitespace before tokenization
    text = normalize_whitespace(text)
    
    # Step 2: Tokenize into sentences
    sentences = tokenize_sentences(text)
    
    # Determine if this is a press release (exception to filtering)
    is_press_release = 'press' in category.lower() or 'press' in title.lower()
    
    # Step 3: Filter by relevance (except press releases)
    sentence_data = []
    for sent in sentences:
        # Only keep if relevant OR if it's a press release, save as dict with following keys:
        if is_relevant_sentence(sent, is_press_release):
            sentence_data.append({
                'date': date,
                'sentence': sent,
                'title': title,
                'category': category,
                'doc_id': doc_id,
                'sentence_length': len(sent.split()),
                'is_press_release': is_press_release
            })
    
    return sentence_data, len(sentences)  # Return total sentence count for stats


def process_file(input_file, output_file):
    # Processes one preprocessed file and writes the sentence-level gold corpus.
    
    file_name = input_file.name
    print(f'\n{file_name}')
    
    # Load document-level data
    df = pd.read_csv(input_file)
    print(f'documents={len(df)}')
    
    # Add doc_id if not present
    if 'doc_id' not in df.columns:
        df['doc_id'] = df.index
    
    # Process each document
    all_sentences = []
    total_sentences_before = 0
    total_sentences_after = 0
    docs_with_relevant_sentences = 0
    
    # iterrows() is slow but sent_tokenize() isn't vectorizable anyway
    for idx, row in df.iterrows():
        sentence_data, total_sent = process_document(
            text=row['text'],
            title=row['title'],
            date=row['date'],
            category=row['category'],
            doc_id=row.get('doc_id', idx)
        )
        
        total_sentences_before += total_sent
        
        if sentence_data:
            all_sentences.extend(sentence_data)
            total_sentences_after += len(sentence_data)
            docs_with_relevant_sentences += 1
    
    # Create sentence-level DataFrame
    df_sentences = pd.DataFrame(all_sentences)
    
    if len(df_sentences) == 0:
        print(f'  WARNING: No relevant sentences found!')
        return None
    
    # Drop very short sentences that slipped through (fragments etc)
    min_words = 3
    initial_count = len(df_sentences)
    df_sentences = df_sentences[df_sentences['sentence_length'] >= min_words]
    removed_short = initial_count - len(df_sentences)
    
    # Sort by date
    df_sentences = df_sentences.sort_values('date').reset_index(drop=True)
    
    # Calculate reduction rate
    reduction_rate = (1 - total_sentences_after / total_sentences_before) * 100 if total_sentences_before > 0 else 0
    
    # Save
    df_sentences.to_csv(output_file, index=False, encoding='utf-8')
    
    # Statistics
    press_release_sentences = df_sentences['is_press_release'].sum()
    filtered_sentences = len(df_sentences) - press_release_sentences
    
    print(f'documents_with_relevant_content={docs_with_relevant_sentences}/{len(df)} ({100*docs_with_relevant_sentences/len(df):.1f}%)')
    print(f'sentences_before_filtering={total_sentences_before:,}')
    print(f'sentences_after_filtering={total_sentences_after:,}')
    print(f'reduction_rate={reduction_rate:.1f}%')
    if press_release_sentences > 0:
        print(f'press_release_sentences={press_release_sentences:,}')
        print(f'filtered_sentences={filtered_sentences:,}')
    if removed_short > 0:
        print(f'removed_short_sentences={removed_short}')
    print(f'final_sentence_count={len(df_sentences):,}')
    print(f'sentence_length_min={df_sentences["sentence_length"].min()}, '
          f'median={df_sentences["sentence_length"].median():.0f}, '
          f'max={df_sentences["sentence_length"].max()}')
    print(f'saved: {output_file.name}')
    
    return df_sentences

def main():
    input_dir = Path('02_Data/preprocessed')
    output_dir = Path('02_Data/gold_corpus')
    output_dir.mkdir(exist_ok=True)
    
    files = [
        'preprocessed_speeches.csv',
        'preprocessed_press_releases.csv',
        'preprocessed_mpr.csv',
        'preprocessed_finstab.csv',
        'preprocessed_banklend.csv'
    ]
    
    total_sentences = 0
    
    for file in files:
        input_path = input_dir / file
        output_name = file.replace('preprocessed_', 'gold_')
        output_path = output_dir / output_name
        
        df_sent = process_file(input_path, output_path)
        
        if df_sent is not None:
            total_sentences += len(df_sent)
    
    print(f'\ntotal_sentences={total_sentences:,}')


if __name__ == '__main__':
    main()
