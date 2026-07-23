import json
import torch
import warnings
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.cuda.amp import autocast
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- config ---
MODEL_DIR  = Path("models/gector_ro_v2/best_model")
VOCAB_PATH = Path("models/gector_ro_v2/best_model/tag_vocabulary.json")

TEST_SOURCES = {
    "cna":        Path("data exploration/errant_input/all/cna/test_source.txt"),
    "rocomments": Path("data exploration/errant_input/all/rocomments/test_source.txt"),
    "rotexts":    Path("data exploration/errant_input/all/rotexts/test_source.txt"),
}

OUTPUT_DIR     = Path("data exploration/errant_output/test/corrected")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH     = 128
BATCH_SIZE     = 64
THRESHOLD      = 0.8
MAX_ITERATIONS = 3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------


def load_vocabulary(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    vocab  = vocab_data['vocab']
    tag2id = {tag: i for i, tag in enumerate(vocab)}
    id2tag = {i: tag for tag, i in tag2id.items()}
    print(f"Loaded vocabulary with {len(vocab)} tags")
    return vocab, tag2id, id2tag


def sanitize(text):
    return (text
            .replace("ţ", "ț").replace("ş", "ș")
            .replace("Ţ", "Ț").replace("Ş", "Ș"))


def get_base_tag(tag):
    if tag in ("$KEEP", "$DELETE"):
        return tag
    if tag.startswith("$APPEND"):
        return "$APPEND"
    if tag.startswith("$REPLACE_SPELL"):
        return "$REPLACE_SPELL"
    if tag.startswith("$REPLACE_VERB"):
        return "$REPLACE_VERB"
    if tag.startswith("$REPLACE_NOUN"):
        return "$REPLACE_NOUN"
    if tag.startswith("$REPLACE_ADJ"):
        return "$REPLACE_ADJ"
    if tag.startswith("$REPLACE"):
        return "$REPLACE"
    return "$KEEP"


def parse_correction(tag):
    base = get_base_tag(tag)
    if base in ("$KEEP", "$DELETE"):
        return base, None
    parts = tag.split("_", 2)
    if base == "$APPEND" and len(parts) >= 2:
        correction = "_".join(parts[1:])
        return "$APPEND", correction
    if len(parts) == 3:
        return base, parts[2]
    return base, None


def predict_tags_batch(sentences, tokenizer, model, id2tag, threshold=0.5):
    all_tokens = [s.split() for s in sentences]

    encoding = tokenizer(
        all_tokens,
        is_split_into_words=True,
        max_length=MAX_LENGTH,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    input_ids      = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        with autocast():
            outputs          = model(input_ids=input_ids, attention_mask=attention_mask)
        probs                = torch.softmax(outputs.logits, dim=-1)
        max_probs, preds     = torch.max(probs, dim=-1)

    results = []
    for batch_idx, tokens in enumerate(all_tokens):
        word_ids          = encoding.word_ids(batch_index=batch_idx)
        token_tags        = []
        previous_word_idx = None

        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue
            if word_idx != previous_word_idx:
                if word_idx < len(tokens):
                    if max_probs[batch_idx][i].item() < threshold:
                        tag = "$KEEP"
                    else:
                        tag = id2tag[preds[batch_idx][i].item()]
                    token_tags.append((tokens[word_idx], tag))
            previous_word_idx = word_idx

        results.append(token_tags)

    return results


def apply_corrections(token_tags):
    corrected = []

    for token, tag in token_tags:
        base, correction = parse_correction(tag)

        if base == "$KEEP":
            corrected.append(token)
        elif base == "$DELETE":
            pass
        elif base == "$APPEND":
            corrected.append(token)
            if correction:
                corrected.append(correction)
        else:
            if correction:
                corrected.append(correction)
            else:
                corrected.append(token)

    return " ".join(corrected)


def correct_sentences_iteratively(sentences, tokenizer, model, id2tag,
                                   threshold=0.5, max_iterations=3):
    current           = list(sentences)
    iteration_results = []

    for iteration in range(max_iterations):
        changed        = False
        next_sentences = []

        for i in range(0, len(current), BATCH_SIZE):
            batch     = current[i:i+BATCH_SIZE]
            tag_preds = predict_tags_batch(batch, tokenizer, model, id2tag, threshold)

            for j, token_tags in enumerate(tag_preds):
                corrected = apply_corrections(token_tags)
                next_sentences.append(corrected)
                if corrected != batch[j]:
                    changed = True

        current = next_sentences
        iteration_results.append(list(current))

        if not changed:
            print(f"  Converged after {iteration + 1} iteration(s)")
            while len(iteration_results) < max_iterations:
                iteration_results.append(list(current))
            break

    return iteration_results


def process_dataset(dataset_name, source_path, tokenizer, model, id2tag):
    print(f"\nProcessing {dataset_name}...")

    with open(source_path, 'r', encoding='utf-8') as f:
        sentences = [sanitize(line.strip()) for line in f if line.strip()]

    print(f"  Loaded {len(sentences)} sentences")

    iteration_results = correct_sentences_iteratively(
        sentences, tokenizer, model, id2tag,
        threshold=THRESHOLD,
        max_iterations=MAX_ITERATIONS
    )

    output_paths = {}
    for i, results in enumerate(iteration_results):
        output_path = OUTPUT_DIR / f"corrected_{dataset_name}_iter{i+1}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in results:
                f.write(sentence + "\n")
        print(f"  Iteration {i+1} written to {output_path}")
        output_paths[i+1] = output_path

    return output_paths


if __name__ == "__main__":
    print(f"Loading model from {MODEL_DIR}...")
    vocab, tag2id, id2tag = load_vocabulary(VOCAB_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded. Device: {DEVICE}")

    all_output_paths = {}
    for dataset_name, source_path in TEST_SOURCES.items():
        if not source_path.exists():
            print(f"Warning: {source_path} not found, skipping...")
            continue
        all_output_paths[dataset_name] = process_dataset(
            dataset_name, source_path, tokenizer, model, id2tag
        )

    print("\nAll done.")
    print("\nOutput files:")
    for dataset, paths in all_output_paths.items():
        for iteration, path in paths.items():
            print(f"  {dataset} iter{iteration}: {path}")