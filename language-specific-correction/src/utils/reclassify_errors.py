import re
from pathlib import Path

# --- Romanian verb form pairs (same lemma, different form) ---
VERB_AGREEMENT_PAIRS = {
    # number agreement (SVA)
    ('a', 'au'), ('au', 'a'),
    ('este', 'sunt'), ('sunt', 'este'),
    ('era', 'erau'), ('erau', 'era'),
    ('a fost', 'au fost'), ('au fost', 'a fost'),
    ('fusese', 'fuseseră'), ('fuseseră', 'fusese'),
    ('face', 'fac'), ('fac', 'face'),
    ('vine', 'vin'), ('vin', 'vine'),
    ('merge', 'merg'), ('merg', 'merge'),
    ('poate', 'pot'), ('pot', 'poate'),
    ('vrea', 'vor'), ('vor', 'vrea'),
    ('știe', 'știu'), ('știu', 'știe'),
    ('are', 'au'), ('au', 'are'),
    ('trebuie', 'trebuiesc'),
}

VERB_TENSE_PAIRS = {
    # tense switches
    ('merge', 'a mers'), ('a mers', 'merge'),
    ('face', 'a făcut'), ('a făcut', 'face'),
    ('vine', 'a venit'), ('a venit', 'vine'),
    ('este', 'a fost'), ('a fost', 'este'),
    ('are', 'a avut'), ('a avut', 'are'),
    ('știe', 'a știut'), ('a știut', 'știe'),
    ('vede', 'a văzut'), ('a văzut', 'vede'),
    ('zice', 'a zis'), ('a zis', 'zice'),
    ('pune', 'a pus'), ('a pus', 'pune'),
    ('vine', 'va veni'), ('va veni', 'vine'),
    ('merge', 'va merge'), ('va merge', 'merge'),
}

PUNCT_CHARS = set('.,;:!?-–—()[]{}"\'\u201e\u201d')

DIACRITIC_MAP = str.maketrans('ăâșțĂÂȘȚ', 'aastAAST')


def strip_diacritics(s):
    return s.translate(DIACRITIC_MAP)


def is_diacritic_fix(orig, cor):
    return strip_diacritics(orig.lower()) == strip_diacritics(cor.lower()) and orig.lower() != cor.lower()


def is_punct_only(orig, cor):
    # one side is empty or both differ only by punctuation
    orig_clean = orig.strip(''.join(PUNCT_CHARS)).strip()
    cor_clean = cor.strip(''.join(PUNCT_CHARS)).strip()
    return orig_clean == cor_clean or orig_clean == '' or cor_clean == ''


def is_case_fix(orig, cor):
    return orig.lower() == cor.lower() and orig != cor


def is_verb_sva(orig, cor):
    return (orig.lower(), cor.lower()) in VERB_AGREEMENT_PAIRS


def is_verb_tense(orig, cor):
    return (orig.lower(), cor.lower()) in VERB_TENSE_PAIRS


def is_verb_form(orig, cor):
    # same stem, both look like verbs — catches things like simplificam -> simplificăm
    if is_diacritic_fix(orig, cor):
        return False  # that's SPELL
    orig_s = strip_diacritics(orig.lower())
    cor_s = strip_diacritics(cor.lower())
    common = sum(1 for a, b in zip(orig_s, cor_s) if a == b)
    ratio = common / max(len(orig_s), len(cor_s)) if max(len(orig_s), len(cor_s)) > 0 else 0
    return ratio > 0.7 and orig_s != cor_s and len(orig) > 2


def is_noun_form(orig, cor):
    # noun declension — similar chars, different ending
    orig_s = strip_diacritics(orig.lower())
    cor_s = strip_diacritics(cor.lower())
    if len(orig_s) < 3 or len(cor_s) < 3:
        return False
    common = sum(1 for a, b in zip(orig_s, cor_s) if a == b)
    ratio = common / max(len(orig_s), len(cor_s))
    return ratio > 0.6 and orig_s != cor_s


def is_whitespace_fix(orig, cor):
    return orig.replace(' ', '') == cor.replace(' ', '') and orig != cor


def reclassify_replace(orig, cor, current_tag):
    """Reclassify R:OTHER into something more specific."""
    if current_tag != 'R:OTHER':
        return current_tag

    orig_c = orig.strip(''.join(PUNCT_CHARS)).strip()
    cor_c = cor.strip(''.join(PUNCT_CHARS)).strip()

    if is_whitespace_fix(orig, cor):
        return 'R:ORTH'
    if is_case_fix(orig_c, cor_c):
        return 'R:ORTH'
    if is_punct_only(orig, cor):
        return 'R:PUNCT'
    if is_diacritic_fix(orig_c, cor_c):
        return 'R:SPELL'
    if is_verb_sva(orig_c, cor_c):
        return 'R:VERB:SVA'
    if is_verb_tense(orig_c, cor_c):
        return 'R:VERB:TENSE'
    if is_verb_form(orig_c, cor_c):
        return 'R:VERB:FORM'
    if is_noun_form(orig_c, cor_c):
        return 'R:NOUN:FORM'

    return current_tag


def reclassify_onesided(orig_or_cor, current_tag):
    """Reclassify M:OTHER and U:OTHER."""
    if not current_tag.endswith(':OTHER'):
        return current_tag

    token = orig_or_cor.strip(''.join(PUNCT_CHARS)).strip()
    op = current_tag[0]  # M or U

    if not token or all(c in PUNCT_CHARS for c in token):
        return f'{op}:PUNCT'
    if token.lower() in {'și', 'sau', 'dar', 'că', 'să', 'ca', 'ori', 'iar'}:
        return f'{op}:CONJ'
    if token.lower() in {'un', 'o', 'unei', 'unui', 'al', 'a', 'ai', 'ale', 'cel', 'cea', 'cei', 'cele'}:
        return f'{op}:DET'
    if token.lower() in {'în', 'la', 'de', 'pe', 'cu', 'din', 'prin', 'pentru', 'despre', 'între', 'până'}:
        return f'{op}:PREP'

    return current_tag


def reclassify_m2(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    sentences = content.strip().split('\n\n')
    output = []
    reclassified = 0
    total = 0

    for sent in sentences:
        lines = sent.strip().split('\n')
        if not lines:
            continue

        source_line = lines[0]
        tokens = source_line[2:].split()

        new_lines = [source_line]
        for line in lines[1:]:
            if not line.startswith('A '):
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
            end = int(span[1])
            current_tag = parts[1]
            correction = parts[2]

            orig = ' '.join(tokens[start:end])
            total += 1

            if current_tag == 'R:OTHER':
                new_tag = reclassify_replace(orig, correction, current_tag)
            elif current_tag in ('M:OTHER', 'U:OTHER'):
                token = correction if current_tag == 'M:OTHER' else orig
                new_tag = reclassify_onesided(token, current_tag)
            else:
                new_tag = current_tag

            if new_tag != current_tag:
                reclassified += 1

            parts[1] = new_tag
            new_lines.append('|||'.join(parts))

        output.append('\n'.join(new_lines))

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(output) + '\n\n')

    print(f"Done. Reclassified {reclassified}/{total} annotations ({100*reclassified/total:.1f}%)")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    reclassify_m2(
        "data exploration/errant_output/annotations_train.m2",
        "data exploration/errant_output/annotations_train_reclassified.m2"
    )