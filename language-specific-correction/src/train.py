import json
import torch
import warnings
import numpy as np
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from seqeval.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import fbeta_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# paths
TRAIN_PATH = Path("data exploration/data/jsonl/train_sequences.jsonl")
VAL_PATH   = Path("data exploration/data/jsonl/val_sequences.jsonl")
VOCAB_PATH = Path("data exploration/data/jsonl/tag_vocabulary.json")
OUTPUT_DIR = Path("models/gector_ro_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# config
MODEL_NAME    = "dumitrescustefan/bert-base-romanian-cased-v1"
MAX_LENGTH    = 128
BATCH_SIZE    = 64
LEARNING_RATE = 1e-5
COLD_LR       = 1e-3
N_EPOCHS      = 10
N_COLD_EPOCHS = 2
PATIENCE      = 3       # early stopping patience
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def resolve_tag(tag, tag2id):
    """Return tag if in vocabulary, otherwise collapse to base tag."""
    if tag in tag2id:
        return tag
    base = get_base_tag(tag)
    return base if base in tag2id else "$KEEP"


def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            item['tokens'] = [sanitize(t) for t in item['tokens']]
            item['source'] = sanitize(item['source'])
            item['target'] = sanitize(item['target'])
            data.append(item)
    print(f"Loaded {len(data)} sequences from {path}")
    return data


def compute_class_weights(data, tag2id, tags):
    counts = Counter()
    for item in data:
        for tag in item['tags']:
            resolved = resolve_tag(tag, tag2id)
            counts[resolved] += 1

    total = sum(counts.values())
    print("\nTop 20 tags by count:")
    weights = []
    for tag in tags:
        count  = counts.get(tag, 1)
        weight = np.sqrt(total / (len(tags) * count))
        weights.append(weight)

    for tag, count in counts.most_common(20):
        idx    = tags.index(tag) if tag in tags else -1
        weight = weights[idx] if idx >= 0 else 0
        print(f"  {tag:<40} count: {count:>10}  weight: {weight:.4f}")

    return torch.tensor(weights, dtype=torch.float)


class WeightedGECLoss(torch.nn.Module):
    def __init__(self, class_weights, ignore_index=-100):
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=ignore_index
        )

    def forward(self, logits, labels):
        return self.loss_fn(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        )


class GECDataset(Dataset):

    def __init__(self, data, tokenizer, tag2id, max_length=128):
        self.data      = data
        self.tokenizer = tokenizer
        self.tag2id    = tag2id
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
                resolved = resolve_tag(tag, self.tag2id)
                aligned_labels.append(self.tag2id.get(resolved, self.tag2id["$KEEP"]))
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        return {
            'input_ids':      encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels':         torch.tensor(aligned_labels, dtype=torch.long)
        }


def compute_f05(all_labels, all_preds, tags):
    flat_labels = [l for seq in all_labels for l in seq]
    flat_preds  = [p for seq in all_preds  for p in seq]

    # per base-class F0.5 — correct if full tag matches
    base_tags = [
        "$DELETE", "$APPEND", "$REPLACE",
        "$REPLACE_SPELL", "$REPLACE_VERB", "$REPLACE_NOUN", "$REPLACE_ADJ"
    ]

    per_class = {}
    for base in base_tags:
        # label is this base class if its base tag matches
        tag_labels = [1 if get_base_tag(l) == base else 0 for l in flat_labels]
        # prediction is correct only if full tag matches gold exactly
        tag_preds  = [1 if (get_base_tag(l) == base and p == l) else 0
                      for l, p in zip(flat_labels, flat_preds)]
        score = fbeta_score(tag_labels, tag_preds, beta=0.5, zero_division=0)
        per_class[base] = score

    # overall — correct only if full tag matches
    overall_labels = [1 if l != "$KEEP" else 0 for l in flat_labels]
    overall_preds  = [1 if (l != "$KEEP" and p == l) else 0
                      for l, p in zip(flat_labels, flat_preds)]
    overall_f05 = fbeta_score(overall_labels, overall_preds, beta=0.5, zero_division=0)

    return overall_f05, per_class


def evaluate(model, dataloader, loss_fn, id2tag, tags):
    model.eval()
    all_preds  = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = loss_fn(outputs.logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=-1)

            for pred_seq, label_seq in zip(preds, labels):
                pred_tags = []
                true_tags = []
                for p, l in zip(pred_seq, label_seq):
                    if l.item() != -100:
                        pred_tags.append(id2tag[p.item()])
                        true_tags.append(id2tag[l.item()])
                all_preds.append(pred_tags)
                all_labels.append(true_tags)

    avg_loss       = total_loss / len(dataloader)
    f1             = f1_score(all_labels, all_preds)
    precision      = precision_score(all_labels, all_preds)
    recall         = recall_score(all_labels, all_preds)
    f05, per_class = compute_f05(all_labels, all_preds, tags)

    return avg_loss, f1, precision, recall, f05, per_class


def train():
    tags, tag2id, id2tag = load_vocabulary(VOCAB_PATH)

    print(f"Using device: {DEVICE}")
    print(f"Vocabulary size: {len(tags)}")

    train_data = load_jsonl(TRAIN_PATH)
    val_data   = load_jsonl(VAL_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = GECDataset(train_data, tokenizer, tag2id, MAX_LENGTH)
    val_dataset   = GECDataset(val_data,   tokenizer, tag2id, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True
    )

    class_weights = compute_class_weights(train_data, tag2id, tags).to(DEVICE)
    loss_fn       = WeightedGECLoss(class_weights)

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(tags),
        id2label=id2tag,
        label2id=tag2id,
        ignore_mismatched_sizes=True
    )
    model.to(DEVICE)

    encoder_params    = list(model.bert.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = AdamW([
        {'params': encoder_params,    'lr': LEARNING_RATE, 'weight_decay': 0.01},
        {'params': classifier_params, 'lr': COLD_LR,       'weight_decay': 0.01}
    ])

    total_steps = len(train_loader) * N_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )

    # mixed precision scaler
    scaler = GradScaler()

    best_f05      = 0
    best_epoch    = 0
    patience_ctr  = 0

    for epoch in range(N_EPOCHS):

        if epoch < N_COLD_EPOCHS:
            print(f"\nEpoch {epoch + 1}/{N_EPOCHS} — COLD (encoder frozen)")
            for param in model.bert.parameters():
                param.requires_grad = False
        else:
            print(f"\nEpoch {epoch + 1}/{N_EPOCHS} — WARM (encoder unfrozen)")
            for param in model.bert.parameters():
                param.requires_grad = True

        model.train()
        total_train_loss = 0

        progress_bar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}")

        for batch in progress_bar:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels         = batch['labels'].to(DEVICE)

            optimizer.zero_grad()

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = loss_fn(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_train_loss / len(train_loader)

        val_loss, val_f1, val_precision, val_recall, val_f05, per_class = evaluate(
            model, val_loader, loss_fn, id2tag, tags
        )

        print(f"\nEpoch {epoch + 1} results:")
        print(f"  Train loss:     {avg_train_loss:.4f}")
        print(f"  Val loss:       {val_loss:.4f}")
        print(f"  Val F1:         {val_f1:.4f}")
        print(f"  Val F0.5:       {val_f05:.4f}")
        print(f"  Val Precision:  {val_precision:.4f}")
        print(f"  Val Recall:     {val_recall:.4f}")
        print(f"  Per-class F0.5:")
        for tag, score in per_class.items():
            print(f"    {tag:<20} {score:.4f}")

        if val_f05 > best_f05:
            best_f05     = val_f05
            best_epoch   = epoch + 1
            patience_ctr = 0
            model.save_pretrained(OUTPUT_DIR / "best_model")
            tokenizer.save_pretrained(OUTPUT_DIR / "best_model")
            # save vocabulary alongside model
            with open(OUTPUT_DIR / "best_model" / "tag_vocabulary.json", 'w', encoding='utf-8') as f:
                json.dump({'vocab': tags, 'tag2id': tag2id, 'id2tag': {str(k): v for k, v in id2tag.items()}}, f, ensure_ascii=False)
            print(f"  ✅ New best model saved (F0.5: {best_f05:.4f})")
        else:
            patience_ctr += 1
            print(f"  No improvement (best F0.5: {best_f05:.4f} at epoch {best_epoch}, patience: {patience_ctr}/{PATIENCE})")
            if patience_ctr >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    print(f"\nTraining complete!")
    print(f"Best F0.5: {best_f05:.4f} at epoch {best_epoch}")
    print(f"Best model saved to {OUTPUT_DIR / 'best_model'}")


if __name__ == "__main__":
    train()