import gzip
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher

# --- config ---
LEXICON_PATH = Path("wfl-ro.txt.gz")

FILES = {
    "train": Path("data exploration/errant_output/annotations_train_reclassified.m2"),
    "val":   Path("data exploration/errant_output/annotations_val_reclassified.m2"),
    "test":  Path("data exploration/errant_output/annotations_test_reclassified.m2"),
}

PUNCT_CHARS = set('.,;:!?-–—()[]{}"\'\u201e\u201d')

DIAC_TABLE = str.maketrans('ăâșțĂÂȘȚîÎ', 'aastAAStiI')

# expanded verb SVA pairs
VERB_SVA_PAIRS = {
    ('a', 'au'), ('au', 'a'),
    ('su', 'au'), ('su', 'și'),
    ('ai', 'au'), ('au', 'ai'),
    ('am', 'a'), ('a', 'am'),
    ('vi', 'fi'), ('ri', 'fi'),
    ('vor', 'va'), ('va', 'vor'),
    ('vz', 'va'), ('vw', 'va'),
    ('s', 'să'), ('sa', 'să'),
    ('z', 'a'), ('w', 'a'),
    ('or', 'vor'), ('u', 'au'),
    ('e', 'este'), ('i', 'și'),
    ('iș', 'și'), ('su', 'se'),
    ('sw', 'se'), ('sw', 'și'),
    ('ș', 'și'),
}
# --------------


def decode_msd(msd):
    if not msd:
        return {}
    cat = msd[0].upper()
    info = {'cat': cat}
    if cat == 'V' and len(msd) >= 3:
        mood_map  = {'i': 'ind', 's': 'subj', 'm': 'imp', 'n': 'inf', 'p': 'part', 'g': 'ger'}
        tense_map = {'p': 'pres', 'i': 'imperf', 's': 'past', 'l': 'pluperf', 'f': 'fut'}
        info['type']   = msd[1] if len(msd) > 1 else None
        info['mood']   = mood_map.get(msd[2], msd[2]) if len(msd) > 2 else None
        info['tense']  = tense_map.get(msd[3], msd[3]) if len(msd) > 3 else None
        info['person'] = msd[4] if len(msd) > 4 else None
        info['number'] = msd[5] if len(msd) > 5 else None
    elif cat == 'N' and len(msd) >= 2:
        case_map = {'n': 'nom', 'g': 'gen', 'd': 'dat', 'a': 'acc', 'v': 'voc', 'r': 'nom/acc', 'y': 'gen/dat'}
        info['gender'] = msd[2] if len(msd) > 2 else None
        info['number'] = msd[3] if len(msd) > 3 else None
        info['case']   = case_map.get(msd[4], msd[4]) if len(msd) > 4 else None
    elif cat == 'A' and len(msd) >= 2:
        info['gender'] = msd[3] if len(msd) > 3 else None
        info['number'] = msd[4] if len(msd) > 4 else None
        info['case']   = msd[5] if len(msd) > 5 else None
    return info


def load_lexicon(path):
    print(f"Loading lexicon from {path}...")
    lexicon = defaultdict(list)
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            word_form, lemma, msd = parts
            info = decode_msd(msd)
            info['lemma'] = lemma
            lexicon[word_form.lower()].append(info)
    print(f"Loaded {sum(len(v) for v in lexicon.values())} entries for {len(lexicon)} word forms")
    return lexicon


def strip_punct(s):
    return s.strip(''.join(PUNCT_CHARS)).strip()


def get_best_entry(lexicon, word):
    # try exact, then stripped punct, then lowercase
    for key in [word.lower(), strip_punct(word).lower()]:
        entries = lexicon.get(key, [])
        if entries:
            return entries[0]
    return None


def is_diacritic_fix(orig, cor):
    o = strip_punct(orig).lower()
    c = strip_punct(cor).lower()
    return o != c and o.translate(DIAC_TABLE) == c.translate(DIAC_TABLE)


def is_short_noise(orig, cor):
    # short garbled token with high char similarity
    o = strip_punct(orig)
    c = strip_punct(cor)
    if len(o) > 4:
        return False
    ratio = SequenceMatcher(None, o.lower(), c.lower()).ratio()
    return ratio >= 0.5 and o.lower() != c.lower()


def is_punct_only_diff(orig, cor):
    return strip_punct(orig).lower() == strip_punct(cor).lower() and orig != cor


def reclassify_other(orig, cor, lexicon):
    op = 'R'
    o_clean = strip_punct(orig)
    c_clean = strip_punct(cor)

    # 1. only punctuation differs
    if is_punct_only_diff(orig, cor):
        return 'R:PUNCT'

    # 2. diacritics fix
    if is_diacritic_fix(orig, cor):
        return 'R:SPELL'

    # 3. short noisy token — likely OCR garbage
    if is_short_noise(orig, cor):
        return 'R:SPELL'

    # 4. verb SVA lookup
    if (o_clean.lower(), c_clean.lower()) in VERB_SVA_PAIRS:
        return 'R:VERB:SVA'

    # 5. lexicon lookup — try with punct stripped
    orig_e = get_best_entry(lexicon, o_clean)
    cor_e  = get_best_entry(lexicon, c_clean)

    if orig_e and cor_e:
        same_lemma = orig_e['lemma'] == cor_e['lemma']
        orig_cat   = orig_e['cat']
        cor_cat    = cor_e['cat']

        if orig_cat == 'V' and cor_cat == 'V':
            if same_lemma:
                if orig_e.get('tense') != cor_e.get('tense') and orig_e.get('tense') and cor_e.get('tense'):
                    return 'R:VERB:TENSE'
                if orig_e.get('mood') != cor_e.get('mood'):
                    return 'R:VERB:FORM'
                if orig_e.get('number') != cor_e.get('number') or orig_e.get('person') != cor_e.get('person'):
                    return 'R:VERB:SVA'
                return 'R:VERB:FORM'
            return 'R:VERB'

        elif orig_cat == 'N' and cor_cat == 'N':
            if same_lemma:
                if orig_e.get('number') != cor_e.get('number'):
                    return 'R:NOUN:NUM'
                if orig_e.get('case') != cor_e.get('case'):
                    return 'R:NOUN:FORM'
                return 'R:NOUN:FORM'
            return 'R:NOUN'

        elif orig_cat == 'A' and cor_cat == 'A':
            if same_lemma:
                return 'R:ADJ:FORM'
            return 'R:ADJ'

        elif orig_cat == 'P' and cor_cat == 'P':
            return 'R:PRON'

        elif orig_cat == 'D' and cor_cat == 'D':
            return 'R:DET'

        elif orig_cat == 'S' and cor_cat == 'S':
            return 'R:PREP'

    elif orig_e and not cor_e:
        if orig_e['cat'] == 'V': return 'R:VERB'
        if orig_e['cat'] == 'N': return 'R:NOUN'
        if orig_e['cat'] == 'A': return 'R:ADJ'

    elif cor_e and not orig_e:
        if cor_e['cat'] == 'V': return 'R:VERB'
        if cor_e['cat'] == 'N': return 'R:NOUN'
        if cor_e['cat'] == 'A': return 'R:ADJ'

    return 'R:OTHER'


def block_has_other(block):
    for line in block.split('\n'):
        if line.startswith('A ') and '|||R:OTHER|||' in line:
            return True
    return False


def reclassify_pass2(input_path, output_path, lexicon):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = [b.strip() for b in content.strip().split('\n\n') if b.strip()]
    output = []
    reclassified = 0
    total_other = 0

    for block in blocks:
        if not block_has_other(block):
            output.append(block)
            continue

        lines = block.split('\n')
        tokens = lines[0][2:].split()
        new_lines = [lines[0]]

        for line in lines[1:]:
            if not line.startswith('A ') or '|||R:OTHER|||' not in line:
                new_lines.append(line)
                continue

            parts = line.split('|||')
            if len(parts) < 3:
                new_lines.append(line)
                continue

            span = parts[0][2:].split()
            if span == ['-1', '-1']:
                new_lines.append(line)
                continue

            start = int(span[0])
            end   = int(span[1])
            orig  = ' '.join(tokens[start:end])
            cor   = parts[2].strip()
            total_other += 1

            new_tag = reclassify_other(orig, cor, lexicon)
            if new_tag != 'R:OTHER':
                reclassified += 1

            parts[1] = new_tag
            new_lines.append('|||'.join(parts))

        output.append('\n'.join(new_lines))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(output) + '\n\n')

    print(f"  R:OTHER processed: {total_other}")
    print(f"  Reclassified:      {reclassified} ({100*reclassified/total_other:.1f}% of OTHER)")
    print(f"  Remaining OTHER:   {total_other - reclassified}")
    print(f"  Written to:        {output_path}")


if __name__ == "__main__":
    lexicon = load_lexicon(LEXICON_PATH)

    for split, input_path in FILES.items():
        output_path = input_path.parent / (input_path.stem.replace("_reclassified", "_reclassified_v2") + ".m2")
        print(f"\nProcessing {input_path.name}...")
        reclassify_pass2(input_path, output_path, lexicon)

    print("\nAll done.")