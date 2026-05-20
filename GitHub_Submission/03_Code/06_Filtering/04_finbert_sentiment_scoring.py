# Scores the gold corpus sentences using FinBERT 
# Same input as LM scoring so the comparison is fair, which I thought was important
# GPU is used if available, on CPU this takes a WHILE with this many sentences.
# Label mapping is 0=neutral, 1=positive, 2=negative in accordance with the documentation 
# Net sentiment computed as = positive - negative
#
# Input:  02_Data/gold_corpus/*.csv
# Output: 02_Data/gold_corpus_finbert/*.csv

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

GOLD_CORPUS_DIR = Path('02_Data/gold_corpus')
OUTPUT_DIR = Path('02_Data/gold_corpus_finbert')
MODEL_NAME = 'yiyanghkust/finbert-tone'

# TEST MODE: Set to True to process only first N sentences per file
TEST_MODE = False
TEST_SAMPLE_SIZE = 100  # Number of sentences to test per file

# RTX 4060 (8GB VRAM): batch_size=16-32 is fine. CPU fallback uses 8.
BATCH_SIZE = 32

# FinBERT label mappings
# VERIFIED: index 0=neutral, 1=positive, 2=negative
LABEL_MAP = {0: 'neutral', 1: 'positive', 2: 'negative'}

# Files to process
GOLD_FILES = [
    'gold_speeches.csv',
    'gold_press_releases.csv',
    'gold_mpr.csv',
    'gold_finstab.csv',
    'gold_banklend.csv'
]


def initialize_finbert():
    # Loads the model and tokenizer, moves to GPU if available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"device={gpu_name} ({gpu_memory:.1f} GB VRAM)")
    else:
        print("device=CPU")
    
    print(f"model={MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    
    model.to(device)
    model.eval()  # disable dropout
    
    return tokenizer, model, device


def score_sentences_batch(sentences, tokenizer, model, device, batch_size=32):
    # Runs sentences through FinBERT in batches and returns scores as a list of dicts.
    results = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        
        # truncate to 512 tokens, pad to equal length within batch
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        inputs = {key: val.to(device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        for j, prob in enumerate(probs):
            neutral_score  = prob[0].item()  # 0=neutral
            positive_score = prob[1].item()  # 1=positive
            negative_score = prob[2].item()  # 2=negative
            net_sentiment = positive_score - negative_score
            predicted_label = LABEL_MAP[torch.argmax(prob).item()]
            
            results.append({
                'finbert_positive_score': positive_score,
                'finbert_negative_score': negative_score,
                'finbert_neutral_score': neutral_score,
                'finbert_net_sentiment': net_sentiment,
                'finbert_predicted_label': predicted_label
            })
    
    return results


def process_gold_corpus_file(filepath, tokenizer, model, device, batch_size=32):
    # Scores all sentences in one corpus file and appends the score columns.
    print(f"\n{filepath.name}")
    
    # Load gold corpus data
    df = pd.read_csv(filepath)
    total_sentences = len(df)
    print(f"sentences={total_sentences:,}")
    
    # Apply test mode if enabled
    if TEST_MODE:
        df = df.head(TEST_SAMPLE_SIZE)
        print(f"test_mode_sentences={len(df)}")
    
    # Extract sentences
    sentences = df['sentence'].tolist()
    
    # Score all sentences with progress bar
    scores = []
    
    for i in tqdm(range(0, len(sentences), batch_size), 
                  desc=f"  {filepath.stem}", 
                  unit="batch"):
        batch = sentences[i:i+batch_size]
        batch_scores = score_sentences_batch(batch, tokenizer, model, device, batch_size=len(batch))
        scores.extend(batch_scores)
    
    # Convert scores to DataFrame
    scores_df = pd.DataFrame(scores)
    
    # Combine with original data
    result = pd.concat([df, scores_df], axis=1)
    
    # Summary statistics
    net = result['finbert_net_sentiment']
    print(f"net_sentiment_mean={net.mean():.4f}, std={net.std():.4f}, min={net.min():.4f}, max={net.max():.4f}")
    
    # Label distribution
    label_counts = result['finbert_predicted_label'].value_counts()
    for label, count in label_counts.items():
        pct = count / len(result) * 100
        print(f"{label}={count:,} ({pct:.1f}%)")
    
    return result


def main():
    if TEST_MODE:
        print(f"test_mode_sample_size={TEST_SAMPLE_SIZE}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize FinBERT model
    tokenizer, model, device = initialize_finbert()
    
    # Adjust batch size for CPU
    batch_size = BATCH_SIZE
    if device.type == 'cpu':
        batch_size = 8
    
    total_sentences = 0
    
    for filename in GOLD_FILES:
        input_path = GOLD_CORPUS_DIR / filename
        
        if not input_path.exists():
            print(f"  {filename} not found, skipping")
            continue
        
        # Process file
        result_df = process_gold_corpus_file(input_path, tokenizer, model, device, batch_size)
        
        # Save to output directory
        output_filename = filename.replace('gold_', 'gold_finbert_')
        output_path = OUTPUT_DIR / output_filename
        result_df.to_csv(output_path, index=False)
        
        print(f"saved: {output_path.name}")
        
        total_sentences += len(result_df)
    
    print(f"\ntotal_sentences={total_sentences:,}")


if __name__ == '__main__':
    main()
