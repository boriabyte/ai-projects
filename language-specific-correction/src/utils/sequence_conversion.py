import pandas as pd
import json
from pathlib import Path

CSV_PATH = Path("data exploration/data/big csvs/big test files/test_all_final.csv")
OUTPUT_PATH = Path("data exploration/data/big csvs/big test files/test_sequences.jsonl")

def convert_to_sequences():
    
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows")
    
    # group by source sentence
    grouped = df.groupby('source', sort=False)
    print(f"Unique sentences: {len(grouped)}")
    
    sequences = []
    skipped = 0
    
    for source, group in grouped:
        
        tokens = source.split()
        n_tokens = len(tokens)
        
        # initialize all tags as $KEEP
        tags = ["$KEEP"] * n_tokens
        
        # apply each error in this sentence
        for _, row in group.iterrows():
            
            start = int(row['start_tok'])
            end = int(row['end_tok'])
            tag = row['tag']
            span_length = end - start
            
            # span length 0 — insertion (M: errors)
            # append after previous token
            if span_length == 0:
                anchor = start - 1
                if anchor >= 0:
                    tags[anchor] = "$APPEND"
                else:
                    skipped += 1
                continue
            
            # skip if out of bounds
            if start >= n_tokens or end > n_tokens:
                skipped += 1
                continue
            
            # span length 1 — single token
            if span_length == 1:
                tags[start] = tag
            
            # span length 2+ — multi token
            elif span_length >= 2:
                tags[start] = tag        # first token gets the operation tag
                for i in range(start + 1, end):
                    if i < n_tokens:
                        tags[i] = "$DELETE"  # remaining tokens get deleted
        
        sequences.append({
            "tokens": tokens,
            "tags": tags,
            "source": source,
            "target": group.iloc[0]['target'],
            "dataset": group.iloc[0]['dataset']
        })
    
    print(f"Skipped {skipped} individual errors due to invalid positions")
    print(f"Generated {len(sequences)} token-tag sequences")
    
    # save as json lines
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for seq in sequences:
            f.write(json.dumps(seq, ensure_ascii=False) + '\n')
    
    print(f"Saved to {OUTPUT_PATH}")
    
    # show a sample
    print(f"\nSample sequence:")
    sample = sequences[0]
    for token, tag in zip(sample['tokens'], sample['tags']):
        if tag != "$KEEP":
            print(f"  {token:20} → {tag}")
        else:
            print(f"  {token:20}   {tag}")
    
    return sequences

if __name__ == "__main__":
    convert_to_sequences()