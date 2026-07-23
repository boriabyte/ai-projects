import json
import torch
import warnings
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import fbeta_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# --- config ---
MODEL_DIR  = Path("models/gector_ro/best_model")
TEST_FILES = {
    "cna":        Path("data exploration/data/jsonl/test/cna_test_sequences.jsonl"),
    "rocomments": Path("data exploration/data/jsonl/test/rocomments_test_sequences.jsonl"),
    "rotexts":    Path("data exploration/data/jsonl/test/rotexts_test_sequences.jsonl"),
}
OUTPUT_DIR = Path("models/gector_ro/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 128
BATCH_SIZE = 32
THRESHOLD  = 0.5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# --------------

TAGS = [
    "$KEEP",
    "$DELETE",
    "$APPEND",
    "$REPLACE",
    "$REPLACE_SPELL",
    "$REPLACE_VERB",
    "$REPLACE_NOUN",
    "$REPLACE_ADJ",
]
TAG2ID = {tag: i for i, tag in enumerate(TAGS)}
ID2TAG = {i: tag for tag, i in TAG2ID.items()}


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


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            item['tokens'] = [sanitize(t) for t in item['tokens']]
            item['source'] = sanitize(item['source'])
            item['target'] = sanitize(item['target'])
            data.append(item)
    print(f"Loaded {len(data)} sequences from {path.name}")
    return data


class GECDataset(Dataset):

    def __init__(self, data, tokenizer, max_length=128):
        self.data       = data
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item   = self.data[idx]
        tokens = item['tokens']
        tags   = item['tags']

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        word_ids          = encoding.word_ids()
        aligned_labels    = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                tag      = tags[word_idx] if word_idx < len(tags) else "$KEEP"
                base_tag = get_base_tag(tag)
                aligned_labels.append(TAG2ID.get(base_tag, TAG2ID["$KEEP"]))
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels':         torch.tensor(aligned_labels, dtype=torch.long)
        }


def compute_f05(all_labels, all_preds):
    flat_labels = [l for seq in all_labels for l in seq]
    flat_preds  = [p for seq in all_preds  for p in seq]

    labels_for_score = [t for t in TAGS if t != "$KEEP"]

    per_class = {}
    for tag in labels_for_score:
        tag_labels = [1 if l == tag else 0 for l in flat_labels]
        tag_preds  = [1 if p == tag else 0 for p in flat_preds]
        score = fbeta_score(tag_labels, tag_preds, beta=0.5, zero_division=0)
        per_class[tag] = score

    overall_labels = [1 if l != "$KEEP" else 0 for l in flat_labels]
    overall_preds  = [1 if p != "$KEEP" else 0 for p in flat_preds]
    overall_f05    = fbeta_score(overall_labels, overall_preds, beta=0.5, zero_division=0)

    return overall_f05, per_class


def evaluate(model, dataloader, threshold=0.5):
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # softmax to get probabilities
            probs              = torch.softmax(outputs.logits, dim=-1)
            max_probs, preds   = torch.max(probs, dim=-1)

            for pred_seq, label_seq, prob_seq in zip(preds, labels, max_probs):
                pred_tags = []
                true_tags = []
                for p, l, prob in zip(pred_seq, label_seq, prob_seq):
                    if l.item() != -100:
                        true_tags.append(ID2TAG[l.item()])
                        # below threshold → fall back to $KEEP
                        if prob.item() < threshold:
                            pred_tags.append("$KEEP")
                        else:
                            pred_tags.append(ID2TAG[p.item()])
                all_preds.append(pred_tags)
                all_labels.append(true_tags)

    f1        = f1_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall    = recall_score(all_labels, all_preds)
    f05, per_class = compute_f05(all_labels, all_preds)

    return f1, precision, recall, f05, per_class


def write_results(output_path, dataset_name, f1, precision, recall, f05, per_class, threshold):
    lines = [
        f"Evaluation results for: {dataset_name}",
        f"Threshold:             {threshold}",
        f"{'='*50}",
        f"",
        f"Overall metrics:",
        f"  F1:        {f1:.4f}",
        f"  F0.5:      {f05:.4f}",
        f"  Precision: {precision:.4f}",
        f"  Recall:    {recall:.4f}",
        f"",
        f"Per-class F0.5:",
    ]
    for tag, score in per_class.items():
        lines.append(f"  {tag:<20} {score:.4f}")

    text = "\n".join(lines)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)


if __name__ == "__main__":
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model     = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded. Using device: {DEVICE}")
    print(f"Confidence threshold: {THRESHOLD}")

    for dataset_name, jsonl_path in TEST_FILES.items():
        print(f"\n{'='*50}")
        print(f"Evaluating {dataset_name}...")

        data       = load_jsonl(jsonl_path)
        dataset    = GECDataset(data, tokenizer, MAX_LENGTH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        f1, precision, recall, f05, per_class = evaluate(model, dataloader, threshold=THRESHOLD)

        output_path = OUTPUT_DIR / f"{dataset_name}_results_threshold_{THRESHOLD}.txt"
        write_results(output_path, dataset_name, f1, precision, recall, f05, per_class, THRESHOLD)
        print(f"\nResults saved to {output_path}")

    print("\nAll evaluations done.")