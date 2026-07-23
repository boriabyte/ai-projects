import json
import random
from pathlib import Path

# --- config ---
M2_FILES = {
    "train": Path("data exploration/errant_output/annotations_train_reclassified_v2.m2"),
    "val":   Path("data exploration/errant_output/annotations_val_reclassified_v2.m2"),
    "test":  Path("data exploration/errant_output/annotations_test_reclassified_v2.m2"),
}

OUTPUT_DIR = Path("data exploration/data/jsonl")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CORRECT_KEEP_RATIO = 0.1  # keep 10% of fully correct sentences
RANDOM_SEED = 42

error_type_to_tag = {
    # keep
    "correct":      "$KEEP",

    # delete
    "U:PUNCT":      "$DELETE",
    "U:NOUN":       "$DELETE",
    "U:VERB":       "$DELETE",
    "U:ADJ":        "$DELETE",
    "U:ADV":        "$DELETE",
    "U:ADP":        "$DELETE",
    "U:SCONJ":      "$DELETE",
    "U:PROPN":      "$DELETE",
    "U:OTHER":      "$DELETE",

    # append
    "M:PUNCT":      "$APPEND",
    "M:NOUN":       "$APPEND",
    "M:VERB":       "$APPEND",
    "M:ADJ":        "$APPEND",
    "M:ADV":        "$APPEND",
    "M:ADP":        "$APPEND",
    "M:SCONJ":      "$APPEND",
    "M:PROPN":      "$APPEND",
    "M:DET":        "$APPEND",
    "M:OTHER":      "$APPEND",

    # specific replaces
    "R:SPELL":      "$REPLACE_SPELL",
    "R:ORTH":       "$REPLACE_SPELL",
    "R:PUNCT":      "$REPLACE_SPELL",
    "R:VERB":       "$REPLACE_VERB",
    "R:VERB:FORM":  "$REPLACE_VERB",
    "R:VERB:TENSE": "$REPLACE_VERB",
    "R:VERB:SVA":   "$REPLACE_VERB",
    "R:NOUN":       "$REPLACE_NOUN",
    "R:NOUN:FORM":  "$REPLACE_NOUN",
    "R:NOUN:NUM":   "$REPLACE_NOUN",
    "R:ADJ":        "$REPLACE_ADJ",
    "R:ADJ:FORM":   "$REPLACE_ADJ",
    "R:PRON":       "$REPLACE",
    "R:DET":        "$REPLACE",
    "R:ADP":        "$REPLACE",
    "R:PREP":       "$REPLACE",
    "R:SCONJ":      "$REPLACE",
    "R:ADV":        "$REPLACE",
    "R:PROPN":      "$REPLACE",
    "R:PROPN:FORM": "$REPLACE",
    "R:MORPH":      "$REPLACE",
    "R:WO":         "$REPLACE",
    "R:OTHER":      "$REPLACE",
}
# --------------


def build_tag(err_type, correction):
    base = error_type_to_tag.get(err_type, "$REPLACE")

    if base == "$KEEP" or base == "$DELETE":
        return base

    correction = correction.strip()
    if correction and correction != "-NONE-":
        return f"{base}_{correction}"

    return base


def reconstruct_target(tokens, edits):
    """Apply edits to source tokens to reconstruct the target sentence."""
    result = list(tokens)
    # sort edits by start position descending to apply without offset issues
    for start, end, err_type, correction in sorted(edits, key=lambda x: x[0], reverse=True):
        if err_type == "noop":
            continue
        base = error_type_to_tag.get(err_type, "$REPLACE")
        cor = correction.strip()
        if base == "$DELETE":
            result[start:end] = []
        elif base == "$APPEND":
            # insert after token at start-1
            insert_pos = start
            result.insert(insert_pos, cor)
        else:
            # replace
            cor_tokens = cor.split()
            result[start:end] = cor_tokens
    return " ".join(result)


def parse_block(block):
    lines = block.strip().split('\n')
    if not lines or not lines[0].startswith('S '):
        return None

    source_line = lines[0][2:].strip()
    tokens = source_line.split()

    edits = []
    for line in lines[1:]:
        if not line.startswith('A '):
            continue
        parts = line.split('|||')
        if len(parts) < 3:
            continue

        span = parts[0][2:].split()
        if len(span) < 2:
            continue

        start = int(span[0])
        end   = int(span[1])
        err_type  = parts[1].strip()
        correction = parts[2].strip()

        if err_type == "noop":
            continue

        edits.append((start, end, err_type, correction))

    return tokens, source_line, edits


def build_tags(tokens, edits):
    tags = ["$KEEP"] * len(tokens)

    for start, end, err_type, correction in edits:
        base = error_type_to_tag.get(err_type, "$REPLACE")

        if base == "$DELETE":
            for i in range(start, min(end, len(tokens))):
                tags[i] = "$DELETE"

        elif base == "$APPEND":
            # assign append tag to token just before insertion point
            insert_at = max(start - 1, 0)
            if insert_at < len(tags):
                cor = correction.strip()
                tags[insert_at] = f"$APPEND_{cor}" if cor and cor != "-NONE-" else "$APPEND"

        else:
            # replace — assign to first token in span
            tag = build_tag(err_type, correction)
            if start < len(tokens):
                tags[start] = tag
            # remaining tokens in multi-token span get $DELETE
            for i in range(start + 1, min(end, len(tokens))):
                tags[i] = "$DELETE"

    return tags


def is_all_keep(tags):
    return all(t == "$KEEP" for t in tags)


def process_m2(m2_path, output_path, keep_ratio=CORRECT_KEEP_RATIO, seed=RANDOM_SEED):
    random.seed(seed)

    with open(m2_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = [b.strip() for b in content.strip().split('\n\n') if b.strip()]

    total = 0
    kept_correct = 0
    dropped_correct = 0
    written = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for block in blocks:
            result = parse_block(block)
            if result is None:
                continue

            tokens, source, edits = result
            tags = build_tags(tokens, edits)
            target = reconstruct_target(tokens, edits)
            total += 1

            # handle fully correct sentences
            if is_all_keep(tags):
                if random.random() > keep_ratio:
                    dropped_correct += 1
                    continue
                kept_correct += 1

            record = {
                "tokens": tokens,
                "tags":   tags,
                "source": source,
                "target": target,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"  Total blocks:       {total}")
    print(f"  Correct kept:       {kept_correct}")
    print(f"  Correct dropped:    {dropped_correct}")
    print(f"  Written to JSONL:   {written}")
    print(f"  Output:             {output_path}")


if __name__ == "__main__":
    for split, m2_path in M2_FILES.items():
        output_path = OUTPUT_DIR / f"{split}_sequences.jsonl"
        print(f"\nProcessing {split} — {m2_path.name}...")
        process_m2(m2_path, output_path)

    print("\nAll done.")