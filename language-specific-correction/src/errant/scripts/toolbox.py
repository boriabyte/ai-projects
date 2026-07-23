from spacy.tokens import Doc

def loadDictionary(path):
    return set(open(path).read().split())

def processM2(info):
    info = info.split("\n")
    orig_sent = info[0][2:].split() # [2:] ignore the leading "S "
    all_edits = info[1:]
    edit_dict = processEdits(all_edits)
    out_dict = {}
    
    for coder, edits in edit_dict.items():
        cor_sent = orig_sent[:]
        gold_edits = []
        offset = 0
        for edit in sorted(edits):
            if edit[2] in {"noop", "Um"}: 
                gold_edits.append(edit + [-1, -1])
                continue
            orig_start = edit[0]
            orig_end = edit[1]
            cor_toks = edit[3].split()
            cor_sent[orig_start+offset:orig_end+offset] = cor_toks
            cor_start = orig_start + offset
            cor_end = cor_start + len(cor_toks)
            offset = offset - (orig_end - orig_start) + len(cor_toks)
            gold_edits.append(edit + [cor_start] + [cor_end])
            
        out_dict[coder] = (cor_sent, gold_edits)
    return orig_sent, out_dict

def processEdits(edits):
    edit_dict = {}
    for edit in edits:
        edit = edit.split("|||")
        span = edit[0][2:].split()
        start = int(span[0])
        end = int(span[1])
        cat = edit[1]
        cor = edit[2]
        id = edit[-1]
        proc_edit = [start, end, cat, cor]
        if id in edit_dict.keys():
            edit_dict[id].append(proc_edit)
        else:
            edit_dict[id] = [proc_edit]
    return edit_dict

def applySpacy(sent, nlp):
    sent = Doc(nlp.vocab, words=sent)
    for name, component in nlp.pipeline:
        sent = component(sent)
    return sent

def minimiseEdit(edit, orig, cor):
    orig_toks = orig[edit[0]:edit[1]]
    cor_toks = cor[edit[4]:edit[5]]
    while orig_toks and cor_toks and orig_toks[0].text == cor_toks[0].text:
        orig_toks = orig_toks[1:]
        cor_toks = cor_toks[1:]
        edit[0] += 1
        edit[4] += 1
    while orig_toks and cor_toks and orig_toks[-1].text == cor_toks[-1].text:
        orig_toks = orig_toks[:-1]
        cor_toks = cor_toks[:-1]
        edit[1] -= 1
        edit[5] -= 1
    if orig_toks or cor_toks:
        edit[3] = " ".join([tok.text for tok in cor_toks])
        return edit	

def formatEdit(edit, coder_id=0):
    span = " ".join(["A", str(edit[0]), str(edit[1])])
    return "|||".join([span, edit[2], edit[3], "REQUIRED", "-NONE-", str(coder_id)])