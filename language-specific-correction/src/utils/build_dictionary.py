from pathlib import Path
import pandas as pd
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

from parse_m2 import parse_m2
from config import *

NUM_WORKERS = 4

def extract_sentence_pairs():
    
    all_sources = []
    all_targets = []
    
    for dataset_name, splits in file_mapping.items():
        for split, filename in splits.items():
            
            file_path = DATA_DIR / dataset_name / filename
            
            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping...")
                continue
            
            df = pd.read_csv(file_path)
            
            if "source" not in df.columns or "target" not in df.columns:
                print(f"Warning: {file_path} missing source/target columns, skipping...")
                continue
            
            df = df.dropna(subset=["source", "target"])
            
            print(f"Loaded {len(df)} pairs from {dataset_name}/{filename}")
            
            dataset_output_dir = OUTPUT_DIR / dataset_name
            dataset_output_dir.mkdir(parents=True, exist_ok=True)
            
            source_file = dataset_output_dir / f"{split}_source.txt"
            target_file = dataset_output_dir / f"{split}_target.txt"
            
            with open(source_file, "w", encoding="utf-8") as f:
                for sentence in df["source"].tolist():
                    f.write(sentence.strip() + "\n")
            
            with open(target_file, "w", encoding="utf-8") as f:
                for sentence in df["target"].tolist():
                    f.write(sentence.strip() + "\n")
            
            print(f"  -> Written to {dataset_output_dir}")
            
            all_sources.extend(df["source"].tolist())
            all_targets.extend(df["target"].tolist())
    
    return all_sources, all_targets


def write_combined_errant_input(sources, targets):
    
    source_file = OUTPUT_DIR / "all_source.txt"
    target_file = OUTPUT_DIR / "all_target.txt"
    
    with open(source_file, "w", encoding="utf-8") as f:
        for sentence in sources:
            f.write(sentence.strip() + "\n")
    
    with open(target_file, "w", encoding="utf-8") as f:
        for sentence in targets:
            f.write(sentence.strip() + "\n")
    
    print(f"\nWrote {len(sources)} total sentence pairs to:")
    print(f"  {source_file}")
    print(f"  {target_file}")
    
    return source_file, target_file


def split_file(lines, n_chunks):
    chunk_size = math.ceil(len(lines) / n_chunks)
    return [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]


def run_errant_chunk(chunk_id, source_lines, target_lines):
    chunk_dir = OUTPUT_DIR_ERR_INF / f"chunk_{chunk_id}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    source_file = chunk_dir / "source.txt"
    target_file = chunk_dir / "target.txt"
    output_file = chunk_dir / "annotations.m2"
    
    with open(source_file, "w", encoding="utf-8") as f:
        f.writelines(source_lines)
    
    with open(target_file, "w", encoding="utf-8") as f:
        f.writelines(target_lines)
    
    result = subprocess.run(
        [
            "python",
            str(ERRANT_DIR / "parallel_to_m2.py"),
            "-orig", str(source_file),
            "-cor", str(target_file),
            "-out", str(output_file)
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Chunk {chunk_id} ERRANT error: {result.stderr}")
        return None
    
    print(f"Chunk {chunk_id} done")
    return output_file


def merge_m2_files(m2_files, output_path):
    with open(output_path, "w", encoding="utf-8") as out:
        for m2_file in m2_files:
            with open(m2_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                out.write(content + "\n\n")
    print(f"Merged {len(m2_files)} chunks into {output_path}")


def deduce_error_type_parallel(source_file, target_file):
    with open(source_file, "r", encoding="utf-8") as f:
        source_lines = f.readlines()
    with open(target_file, "r", encoding="utf-8") as f:
        target_lines = f.readlines()
    
    source_chunks = split_file(source_lines, NUM_WORKERS)
    target_chunks = split_file(target_lines, NUM_WORKERS)
    
    print(f"Split into {len(source_chunks)} chunks of ~{len(source_chunks[0])} lines each")
    
    m2_files = [None] * len(source_chunks)
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(run_errant_chunk, i, source_chunks[i], target_chunks[i]): i
            for i in range(len(source_chunks))
        }
        for future in as_completed(futures):
            chunk_id = futures[future]
            m2_files[chunk_id] = future.result()
    
    if any(f is None for f in m2_files):
        print("Some chunks failed, check errors above")
        return None
    
    final_output = OUTPUT_DIR_ERR_INF / "annotations.m2"
    merge_m2_files(m2_files, final_output)
    return final_output


def append_error_type_train(m2_df):
    df = pd.read_csv(BIG_CSV_TEST)
    
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    
    if "err_type" in df.columns:
        df = df.drop(columns=["err_type"])
    
    print(f"Loaded {len(df)} rows from big CSV")
    
    df = df.merge(m2_df, on="source", how="left")
    
    df["err_type"] = df["err_type"].fillna("correct")
    df["correction"] = df["correction"].fillna("")
    df["start_tok"] = df["start_tok"].fillna(-1).astype(int)
    df["end_tok"] = df["end_tok"].fillna(-1).astype(int)
    
    df.to_csv(OUTPUT_FILE_TEST, index=False)
    print(f"Saved annotated CSV with {len(df)} rows to {OUTPUT_FILE_TEST}")
    print(f"\nSample:\n{df.head(10)}")
    
    return df


def build_full_tag(row):
    
    base_tag = error_type_to_tag.get(row['err_type'], "$REPLACE")
    
    if base_tag == "$KEEP":
        return "$KEEP"
    if base_tag == "$DELETE":
        return "$DELETE"
    
    correction = str(row['correction']).strip()
    if correction:
        return f"{base_tag}_{correction}"
    
    return base_tag


def apply_tag_mapping():
    
    df = pd.read_csv(CSV_PATH_TEST)
    print(f"Loaded {len(df)} rows")
    
    df['full_tag'] = df.apply(build_full_tag, axis=1)
    
    tag_counts = df['full_tag'].value_counts()
    rare_tags = set(tag_counts[tag_counts < MIN_COUNT].index)
    
    print(f"Total unique tags before filtering: {tag_counts.shape[0]}")
    print(f"Rare tags (< {MIN_COUNT} occurrences): {len(rare_tags)}")
    print(f"Tags kept as-is: {tag_counts.shape[0] - len(rare_tags)}")
    
    def finalize_tag(tag):
        if tag in rare_tags:
            if tag.startswith("$REPLACE"):
                return "$REPLACE"
            elif tag.startswith("$APPEND"):
                return "$APPEND"
            elif tag.startswith("$MORPH"):
                return "$MORPH"
        return tag
    
    df['full_tag'] = df['full_tag'].apply(finalize_tag)
    
    print(f"Final vocabulary size: {df['full_tag'].nunique()}")
    print(f"\nTop 20 most common tags:")
    print(df['full_tag'].value_counts().head(20))
    
    df.to_csv(OUTPUT_PATH_TEST, index=False)
    print(f"\nSaved to {OUTPUT_PATH_TEST}")
    
    return df


if __name__ == "__main__":
    sources, targets = extract_sentence_pairs()
    sources_all, targets_all = write_combined_errant_input(sources, targets)
    deduce_error_type_parallel(sources_all, targets_all)
    #m2_df = parse_m2(M2_FILE)
    #append_error_type_train(m2_df)
    #apply_tag_mapping()