from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

from config import *

NUM_WORKERS = 4

DATASETS = {
    "cna":        Path("data exploration/errant_input/all/cna/test_source.txt"),
    "rocomments": Path("data exploration/errant_input/all/rocomments/test_source.txt"),
    "rotexts":    Path("data exploration/errant_input/all/rotexts/test_source.txt"),
}

CORRECTED_DIR = Path("data exploration/errant_output/test/corrected")
PREDICTED_DIR = Path("data exploration/errant_output/test/predicted")
PREDICTED_DIR.mkdir(parents=True, exist_ok=True)

MAX_ITERATIONS = 3


def split_lines(lines, n_chunks):
    chunk_size = math.ceil(len(lines) / n_chunks)
    return [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]


def run_errant_chunk(chunk_id, source_lines, corrected_lines, chunk_base_dir):
    chunk_dir = chunk_base_dir / f"chunk_{chunk_id}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    source_file    = chunk_dir / "source.txt"
    corrected_file = chunk_dir / "corrected.txt"
    output_file    = chunk_dir / "predicted.m2"

    with open(source_file, "w", encoding="utf-8") as f:
        f.writelines(source_lines)
    with open(corrected_file, "w", encoding="utf-8") as f:
        f.writelines(corrected_lines)

    result = subprocess.run(
        [
            "python",
            str(ERRANT_DIR / "parallel_to_m2.py"),
            "-orig", str(source_file),
            "-cor",  str(corrected_file),
            "-out",  str(output_file),
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"  Chunk {chunk_id} ERRANT error: {result.stderr}")
        return None

    print(f"  Chunk {chunk_id} done")
    return output_file


def merge_m2_files(m2_files, output_path):
    with open(output_path, "w", encoding="utf-8") as out:
        for m2_file in m2_files:
            with open(m2_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            out.write(content + "\n\n")
    print(f"  Merged into {output_path.name}")


def run_errant_parallel(source_path, corrected_path, output_m2, chunk_base_dir):
    with open(source_path, "r", encoding="utf-8") as f:
        source_lines = f.readlines()
    with open(corrected_path, "r", encoding="utf-8") as f:
        corrected_lines = f.readlines()

    if len(source_lines) != len(corrected_lines):
        print(f"  WARNING: line count mismatch — source {len(source_lines)}, corrected {len(corrected_lines)}")
        return None

    source_chunks    = split_lines(source_lines, NUM_WORKERS)
    corrected_chunks = split_lines(corrected_lines, NUM_WORKERS)

    m2_files = [None] * len(source_chunks)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(run_errant_chunk, i, source_chunks[i], corrected_chunks[i], chunk_base_dir): i
            for i in range(len(source_chunks))
        }
        for future in as_completed(futures):
            chunk_id = futures[future]
            m2_files[chunk_id] = future.result()

    if any(f is None for f in m2_files):
        print("  Some chunks failed")
        return None

    merge_m2_files(m2_files, output_m2)
    return output_m2


if __name__ == "__main__":
    for dataset_name, source_path in DATASETS.items():
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")

        for iteration in range(1, MAX_ITERATIONS + 1):
            corrected_path = CORRECTED_DIR / f"corrected_{dataset_name}_iter{iteration}.txt"
            output_m2      = PREDICTED_DIR / f"predicted_{dataset_name}_iter{iteration}.m2"
            chunk_dir      = PREDICTED_DIR / "chunks" / dataset_name / f"iter{iteration}"

            if not corrected_path.exists():
                print(f"  Missing corrected file for iter {iteration}, skipping...")
                continue

            print(f"\n  Iteration {iteration}...")
            result = run_errant_parallel(source_path, corrected_path, output_m2, chunk_dir)

            if result:
                print(f"  ✅ Written to {output_m2}")
            else:
                print(f"  ❌ Failed")

    print("\nAll done.")