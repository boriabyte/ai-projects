
from pathlib import Path

dataset_names = ["cna", "rocomments", "ronacc", "rotexts"]

file_mapping = {
    "cna": {"train": "train.csv"},
    "rocomments": {"train": "train.csv"},
    # "ronacc": {"train": "train.csv", "val": "val.csv", "test": "test.csv"},
    "rotexts": {"train": "train.csv"}
}

errant_input_mapping ={
    "cna": {"correct": "train_source.txt", "incorrect": "train_target.txt"},
    "rocomments": {"correct": "train_source.txt", "incorrect": "train_target.txt"},
    # "ronacc": {"train": "train.csv", "val": "val.csv", "test": "test.csv"},
    "rotexts": {"correct": "train_source.txt", "incorrect": "train_target.txt"}
}

DATA_DIR = Path("data exploration/data")
OUTPUT_DIR = Path("data exploration/errant_input/all")
OUTPUT_DIR_ERR_INF = Path("data exploration/errant_output") # should be an m2 file
ERRANT_DIR = Path("errant")

BIG_CSV = Path("data exploration/data/big csvs/train_all.csv")
BIG_CSV_VAL = Path("data exploration/data/big csvs/val_all.csv")
BIG_CSV_TEST = Path("data exploration/data/big csvs/big test files/test_all.csv")
OUTPUT_FILE = Path("data exploration/data/big csvs/train_all_annotated.csv")
OUTPUT_FILE_VAL = Path("data exploration/data/big csvs/VAL_all_annotated.csv")
OUTPUT_FILE_TEST = Path("data exploration/data/big csvs/big test files/test_all_annotated.csv")
M2_FILE = Path("data exploration/errant_output/annotations_test.m2")

CSV_PATH = Path("data exploration/data/big csvs/train_all_annotated.csv")
CSV_PATH_VAL = Path("data exploration/data/big csvs/VAL_all_annotated.csv")
CSV_PATH_TEST = Path("data exploration/data/big csvs/big test files/test_all_annotated.csv")
OUTPUT_PATH = Path("data exploration/data/big csvs/train_all_tagged.csv")
OUTPUT_PATH_VAL = Path("data exploration/data/big csvs/val_all_tagged.csv")
OUTPUT_PATH_TEST = Path("data exploration/data/big csvs/big test files/test_all_tagged.csv")

error_type_to_tag = {
    # no error
    "correct": "$KEEP",
    
    # REPLACE operations
    "R:SPELL":      "$REPLACE",
    "R:ORTH":       "$REPLACE",
    "R:OTHER":      "$REPLACE",
    "R:PROPN":      "$REPLACE",
    "R:DET":        "$REPLACE",
    "R:VERB:TENSE": "$REPLACE",
    "R:ADJ":        "$REPLACE",
    "R:ADV":        "$REPLACE",
    "R:PUNCT":      "$REPLACE",
    "R:NOUN":       "$REPLACE",
    "R:WO":         "$REPLACE",
    
    # MORPHOLOGICAL
    "R:MORPH":      "$MORPH",
    
    # MISSING operations
    "M:OTHER":      "$APPEND",
    "M:DET":        "$APPEND",
    "M:VERB:TENSE": "$APPEND",
    "M:PUNCT":      "$APPEND",
    "M:PROPN":      "$APPEND",
    "M:ADV":        "$APPEND",
    "M:ADJ":        "$APPEND",
    "M:NOUN":       "$APPEND",
    
    # UNNECESSARY operations
    "U:OTHER":      "$DELETE",
    "U:PROPN":      "$DELETE",
    "U:DET":        "$DELETE",
    "U:ADV":        "$DELETE",
    "U:ADJ":        "$DELETE",
    "U:VERB:TENSE": "$DELETE",
    "U:PUNCT":      "$DELETE",
    "U:NOUN":       "$DELETE",
}

MIN_COUNT = 5