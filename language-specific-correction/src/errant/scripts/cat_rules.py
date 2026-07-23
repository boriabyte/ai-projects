from difflib import SequenceMatcher
import spacy.parts_of_speech as spos

# Rare POS tags that make uninformative error categories
rare_tags = {"INTJ", "NUM", "SYM", "X"}
# POS tags with inflectional morphology in Romanian
inflected_tags = {"ADJ", "ADV", "AUX", "DET", "PRON", "PROPN", "NOUN", "VERB"}

# Standard Universal Dependency mapping for specific pos tags.
dep_map = {
    "case": "ADP",   # Prepositions usually have the 'case' dependency
    "mark": "SCONJ", # Conjunctions/infinitive markers
    "punct": "PUNCT" 
}

def autoTypeEdit(edit, orig_sent, cor_sent, word_list, tag_map, nlp, stemmer):
    orig_toks = orig_sent[edit[0]:edit[1]]
    cor_toks = cor_sent[edit[4]:edit[5]]
    
    if not orig_toks and not cor_toks:
        return "UNK"
    elif not orig_toks and cor_toks:
        op = "M:"
        cat = getOneSidedType(cor_toks)
    elif orig_toks and not cor_toks:
        op = "U:"
        cat = getOneSidedType(orig_toks)
    else:
        if orig_toks.text == cor_toks.text:
            return "UNK"
        elif orig_toks[-1].lower_ == cor_toks[-1].lower_ and (len(orig_toks) > 1 or len(cor_toks) > 1):
            min_edit = edit[:]
            min_edit[1] -= 1
            min_edit[5] -= 1
            return autoTypeEdit(min_edit, orig_sent, cor_sent, word_list, tag_map, nlp, stemmer)
        else:
            op = "R:"
            cat = getTwoSidedType(orig_toks, cor_toks, word_list, nlp, stemmer)
    return op + cat


def getEditInfo(toks):
    text = []
    pos = []
    dep = []
    for tok in toks:
        text.append(tok.text)
        pos.append(tok.pos_) # Universally Native POS tags
        dep.append(tok.dep_)
    return text, pos, dep


def getOneSidedType(toks):
    str_list, pos_list, dep_list = getEditInfo(toks)

    if len(set(pos_list)) == 1 and pos_list[0] not in rare_tags:
        return pos_list[0]
    if len(set(dep_list)) == 1 and dep_list[0] in dep_map.keys():
        return dep_map[dep_list[0]]
    if set(pos_list) == {"PART", "VERB"}:
        return "VERB"
    else:
        return "OTHER"


def getTwoSidedType(orig_toks, cor_toks, word_list, nlp, stemmer):
    orig_str, orig_pos, orig_dep = getEditInfo(orig_toks)
    cor_str, cor_pos, cor_dep = getEditInfo(cor_toks)

    # Orthography; i.e. whitespace and/or case errors.
    if onlyOrthChange(orig_str, cor_str):
        return "ORTH"
    # Word Order; only matches exact reordering.
    if exactReordering(orig_str, cor_str):
        return "WO"

    # 1:1 replacements
    if len(orig_str) == len(cor_str) == 1:
        
        # 1. SPELLING AND INFLECTION
        if orig_str[0].isalpha():
            if orig_str[0] not in word_list and orig_str[0].lower() not in word_list:
                if sameLemma(orig_toks[0], cor_toks[0], nlp):
                    pass # Skip to morphology
                else:
                    char_ratio = SequenceMatcher(None, orig_str[0], cor_str[0]).ratio()
                    if char_ratio > 0.5:
                        return "SPELL"
                    else:
                        if orig_pos == cor_pos and orig_pos[0] not in rare_tags:
                            return orig_pos[0]
                        else:
                            return "OTHER"

        # 2. MORPHOLOGY
        if sameLemma(orig_toks[0], cor_toks[0], nlp) and orig_pos[0] in inflected_tags and cor_pos[0] in inflected_tags:
            if orig_pos == cor_pos:
                if orig_pos[0] in inflected_tags:
                    return orig_pos[0] + ":FORM"
            if cor_toks[0].pos_ == "VERB":
                return "VERB:FORM"
            else:
                return "MORPH"

        # 3. Derivational morphology
        if stemmer.stem(orig_str[0]) == stemmer.stem(cor_str[0]) and orig_pos[0] in inflected_tags and cor_pos[0] in inflected_tags:
            return "MORPH"

        # 4. GENERAL
        if orig_pos == cor_pos and orig_pos[0] not in rare_tags:
            return orig_pos[0]
        if orig_dep == cor_dep and orig_dep[0] in dep_map.keys():
            return dep_map[orig_dep[0]]
        
        # Prepositions
        if set(orig_pos + cor_pos) == {"ADP"} or set(orig_dep + cor_dep) == {"case"}:
            return "ADP"
            
        # DET vs PRON Resolution using Romanian Universal Dependencies
        if set(orig_pos + cor_pos) == {"DET", "PRON"}:
            # Subjects and objects are generally pronouns.
            if cor_dep[0] in {"nsubj", "nsubj:pass", "obj", "iobj"}:
                return "PRON"
            # Modifiers or determiners
            if cor_dep[0] in {"nmod", "nmod:poss", "det"}:
                return "DET"
        else:
            return "OTHER"

    # Multi-token replacements
    if len(set(orig_pos + cor_pos)) == 1 and orig_pos[0] not in rare_tags:
        return orig_pos[0]
        
    if len(set(orig_dep + cor_dep)) == 1 and orig_dep[0] in dep_map.keys():
        return dep_map[orig_dep[0]]
        
    # Infinitives / Phrasal verbs.
    if set(orig_pos + cor_pos) == {"PART", "VERB"}:
        if sameLemma(orig_toks[-1], cor_toks[-1], nlp):
            return "VERB:FORM"
        else:
            return "VERB"

    return "OTHER"


def onlyOrthChange(orig_str, cor_str):
    return "".join(orig_str).lower() == "".join(cor_str).lower()


def exactReordering(orig_str, cor_str):
    return sorted([tok.lower() for tok in orig_str]) == sorted([tok.lower() for tok in cor_str])


def sameLemma(orig_tok, cor_tok, nlp):
    return orig_tok.lemma_ == cor_tok.lemma_