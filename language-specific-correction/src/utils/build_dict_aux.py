from pathlib import Path
import pandas as pd
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

from config import *

NUM_WORKERS = 4

GEC_PAIRS_CSV = Path("data exploration/data/gec-pairs-steno/gec-pairs.csv")
STENO_CSV = Path("data exploration/data/gec-pairs-steno/steno_speaker_sentences-gec-2.csv")

ALL_SOURCE      = OUTPUT_DIR / "all_source.txt"
ALL_TARGET      = OUTPUT_DIR / "all_target.txt"
ALL_SOURCE_VAL  = OUTPUT_DIR / "all_source_val.txt"
ALL_TARGET_VAL  = OUTPUT_DIR / "all_target_val.txt"
ALL_SOURCE_TEST = OUTPUT_DIR / "all_source_test.txt"
ALL_TARGET_TEST = OUTPUT_DIR / "all_target_test.txt"

ANNOTATIONS_MAIN  = OUTPUT_DIR_ERR_INF / "annotations.m2"
ANNOTATIONS_TRAIN = OUTPUT_DIR_ERR_INF / "annotations_extended.m2"
ANNOTATIONS_VAL   = OUTPUT_DIR_ERR_INF / "annotations_extended_val.m2"
ANNOTATIONS_TEST  = OUTPUT_DIR_ERR_INF / "annotations_extended_test.m2"


def load_and_split(csv_path):
    df = pd.read_csv(csv_path)

    if "source" not in df.columns or "target" not in df.columns:
        raise ValueError(f"{csv_path} missing source/target columns")

    df = df.dropna(subset=["source", "target"]).reset_index(drop=True)
    print(f"Loaded {len(df)} pairs from {csv_path.name}")

    n = len(df)
    n_train = int(n * 0.7)
    n_val   = int(n * 0.2)

    train = df.iloc[:n_train]
    val   = df.iloc[n_train:n_train + n_val]
    test  = df.iloc[n_train + n_val:]

    print(f"  -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


def append_to_txt(sources, targets, source_file, target_file):
    with open(source_file, "a", encoding="utf-8") as f:
        for s in sources:
            f.write(s.strip() + "\n")
    with open(target_file, "a", encoding="utf-8") as f:
        for t in targets:
            f.write(t.strip() + "\n")
    print(f"Appended {len(sources)} pairs to {source_file.name} and {target_file.name}")


def write_txt(sources, targets, source_file, target_file):
    with open(source_file, "w", encoding="utf-8") as f:
        for s in sources:
            f.write(s.strip().replace("\n", " ") + "\n")
    with open(target_file, "w", encoding="utf-8") as f:
        for t in targets:
            f.write(t.strip().replace("\n", " ") + "\n")
    print(f"Wrote {len(sources)} pairs to {source_file.name} and {target_file.name}")


def split_lines(lines, n_chunks):
    chunk_size = math.ceil(len(lines) / n_chunks)
    return [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]


def run_errant_chunk(chunk_id, source_lines, target_lines, chunk_base_dir):
    chunk_dir = chunk_base_dir / f"chunk_{chunk_id}"
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
            "-cor",  str(target_file),
            "-out",  str(output_file)
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


def run_errant_parallel(source_file, target_file, output_m2, chunk_base_dir):
    with open(source_file, "r", encoding="utf-8") as f:
        source_lines = f.readlines()
    with open(target_file, "r", encoding="utf-8") as f:
        target_lines = f.readlines()

    source_chunks = split_lines(source_lines, NUM_WORKERS)
    target_chunks = split_lines(target_lines, NUM_WORKERS)

    print(f"Split into {len(source_chunks)} chunks of ~{len(source_chunks[0])} lines each")

    m2_files = [None] * len(source_chunks)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(run_errant_chunk, i, source_chunks[i], target_chunks[i], chunk_base_dir): i
            for i in range(len(source_chunks))
        }
        for future in as_completed(futures):
            chunk_id = futures[future]
            m2_files[chunk_id] = future.result()

    if any(f is None for f in m2_files):
        print("Some chunks failed, check errors above")
        return None

    merge_m2_files(m2_files, output_m2)
    return output_m2


def append_m2(source_m2, dest_m2):
    with open(source_m2, "r", encoding="utf-8") as f:
        content = f.read().strip()
    with open(dest_m2, "a", encoding="utf-8") as f:
        f.write("\n\n" + content + "\n\n")
    print(f"Appended {source_m2.name} to {dest_m2.name}")


if __name__ == "__main__":

    # --- load and split both CSVs ---
    gec_train, gec_val, gec_test       = load_and_split(GEC_PAIRS_CSV)
    steno_train, steno_val, steno_test = load_and_split(STENO_CSV)

    # combine splits across both CSVs
    train_sources = gec_train["source"].tolist() + steno_train["source"].tolist()
    train_targets = gec_train["target"].tolist() + steno_train["target"].tolist()

    val_sources   = gec_val["source"].tolist()   + steno_val["source"].tolist()
    val_targets   = gec_val["target"].tolist()   + steno_val["target"].tolist()

    test_sources  = gec_test["source"].tolist()  + steno_test["source"].tolist()
    test_targets  = gec_test["target"].tolist()  + steno_test["target"].tolist()

    # --- append to txt files ---
    append_to_txt(train_sources, train_targets, ALL_SOURCE, ALL_TARGET)
    append_to_txt(val_sources, val_targets, ALL_SOURCE_VAL, ALL_TARGET_VAL)
    append_to_txt(test_sources, test_targets, ALL_SOURCE_TEST, ALL_TARGET_TEST)

    # --- run ERRANT on train split → annotations_extended.m2 ---
    tmp_train_src = OUTPUT_DIR_ERR_INF / "extended_train_source.txt"
    tmp_train_tgt = OUTPUT_DIR_ERR_INF / "extended_train_target.txt"
    write_txt(train_sources, train_targets, tmp_train_src, tmp_train_tgt)

    run_errant_parallel(
        tmp_train_src, tmp_train_tgt,
        ANNOTATIONS_TRAIN,
        OUTPUT_DIR_ERR_INF / "chunks_train"
    )

    # --- append annotations_extended.m2 to annotations.m2 ---
    append_m2(ANNOTATIONS_TRAIN, ANNOTATIONS_MAIN)

    # --- run ERRANT on val split → annotations_extended_val.m2 ---
    tmp_val_src = OUTPUT_DIR_ERR_INF / "extended_val_source.txt"
    tmp_val_tgt = OUTPUT_DIR_ERR_INF / "extended_val_target.txt"
    write_txt(val_sources, val_targets, tmp_val_src, tmp_val_tgt)

    run_errant_parallel(
        tmp_val_src, tmp_val_tgt,
        ANNOTATIONS_VAL,
        OUTPUT_DIR_ERR_INF / "chunks_val"
    )

    # --- run ERRANT on test split → annotations_extended_test.m2 ---
    tmp_test_src = OUTPUT_DIR_ERR_INF / "extended_test_source.txt"
    tmp_test_tgt = OUTPUT_DIR_ERR_INF / "extended_test_target.txt"
    write_txt(test_sources, test_targets, tmp_test_src, tmp_test_tgt)

    run_errant_parallel(
        tmp_test_src, tmp_test_tgt,
        ANNOTATIONS_TEST,
        OUTPUT_DIR_ERR_INF / "chunks_test"
    )

    print("\nAll done.")
    print(f"  Train annotations appended to : {ANNOTATIONS_MAIN}")
    print(f"  Val annotations               : {ANNOTATIONS_VAL}")
    print(f"  Test annotations              : {ANNOTATIONS_TEST}")