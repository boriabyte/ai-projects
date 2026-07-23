import json
import torch
import warnings
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.cuda.amp import autocast

warnings.filterwarnings("ignore")

# --- config ---
MODEL_DIR  = Path("models/gector_ro_v2/best_model")
VOCAB_PATH = Path("models/gector_ro_v2/best_model/tag_vocabulary.json")
THRESHOLD      = 0.7
MAX_ITERATIONS = 3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------


def load_vocabulary(vocab_path):
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    vocab  = vocab_data['vocab']
    tag2id = {tag: i for i, tag in enumerate(vocab)}
    id2tag = {i: tag for tag, i in tag2id.items()}
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
    
    # handle $APPEND
    if base == "$APPEND":
        prefix = "$APPEND_"
        if tag.startswith(prefix) and len(tag) > len(prefix):
            return "$APPEND", tag[len(prefix):]
        return "$APPEND", None

    # handle $REPLACE variants — correction is everything after the base tag
    prefix = base + "_"
    if tag.startswith(prefix) and len(tag) > len(prefix):
        return base, tag[len(prefix):]
    
    return base, None


def predict_tags(sentence, tokenizer, model, id2tag, threshold):
    tokens = sentence.split()

    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=128,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    input_ids      = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)

    with torch.no_grad():
        with autocast():
            outputs          = model(input_ids=input_ids, attention_mask=attention_mask)
        probs                = torch.softmax(outputs.logits, dim=-1).squeeze()
        max_probs, preds     = torch.max(probs, dim=-1)

    word_ids          = encoding.word_ids()
    token_tags        = []
    previous_word_idx = None

    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if word_idx != previous_word_idx:
            if word_idx < len(tokens):
                if max_probs[i].item() < threshold:
                    tag = "$KEEP"
                else:
                    tag = id2tag[preds[i].item()]
                token_tags.append((tokens[word_idx], tag, max_probs[i].item()))
        previous_word_idx = word_idx

    return token_tags


def apply_corrections(token_tags):
    corrected = []
    for token, tag, conf in token_tags:
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
            corrected.append(correction if correction else token)
    return " ".join(corrected)


def correct(sentence, tokenizer, model, id2tag, threshold, max_iterations, verbose):
    sentence = sanitize(sentence)
    current  = sentence

    if verbose:
        print(f"\n{'─'*60}")
        print(f"INPUT:  {sentence}")
        print(f"{'─'*60}")

    for i in range(max_iterations):
        token_tags = predict_tags(current, tokenizer, model, id2tag, threshold)

        if verbose:
            print(f"\nIteration {i+1}:")
            print(f"  {'Token':<25} {'Tag':<30} {'Conf':>6}")
            print(f"  {'─'*63}")
            for token, tag, conf in token_tags:
                marker = "→" if tag != "$KEEP" else " "
                print(f"  {marker} {token:<23} {tag:<30} {conf:>6.3f}")

        corrected = apply_corrections(token_tags)

        if corrected == current:
            if verbose:
                print(f"\n  Converged after {i+1} iteration(s)")
            break

        current = corrected

        if verbose:
            print(f"\n  Result: {current}")

    return current


if __name__ == "__main__":
    print("Loading model...")
    vocab, tag2id, id2tag = load_vocabulary(VOCAB_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded. Device: {DEVICE}")
    print(f"Threshold: {THRESHOLD} | Max iterations: {MAX_ITERATIONS}")
    print("\nType a Romanian sentence and press Enter to correct it.")
    print("Type 'quit' to exit.\n")

    while True:
        sentence = input(">> ").strip()

        if not sentence:
            continue
        if sentence.lower() == "quit":
            print("Bye!")
            break

        result = correct(
            sentence,
            tokenizer, model, id2tag,
            threshold=THRESHOLD,
            max_iterations=MAX_ITERATIONS,
            verbose=True
        )

        print(f"\n{'='*60}")
        print(f"OUTPUT: {result}")
        print(f"{'='*60}")