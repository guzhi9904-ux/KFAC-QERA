"""One prefix per unseen train article, with explicit historical exclusion."""
import hashlib
import re
import struct
from pathlib import Path
import torch
from common import PLAN, read, save_tensors, save_json, sha_file, mo, seed, require, digest


def articles(rows):
    # Exact frozen Exp-3 parser and newline join.
    start = 0; parts = []; title = 'preamble'
    for i, row in enumerate(rows):
        text = row['text']; line = text.strip()
        if re.fullmatch(r'= [^=].*[^=] =', line):
            if parts: yield start, i, title, '\n'.join(parts)
            start = i; parts = []; title = line
        parts.append(text)
    if parts: yield start, len(rows), title, '\n'.join(parts)


def spans(tokens, width=64):
    # Exact byte strings, not collision-prone rolling hashes. ~130 MiB for 256 old windows.
    raw = struct.pack('<'+'I'*len(tokens), *tokens)
    return {raw[4*i:4*(i+width)] for i in range(len(tokens)-width+1)}


def select_articles(candidates, forbidden, titles, texts, n_search=32, n_test=16, length=2048):
    accepted = []; rejected = []; selected_spans = set(); selected_titles = set(); selected_texts = set()
    for row, tokens in sorted(candidates, key=lambda item: (seed('article-order', item[0]['article_id'], 0), item[0]['article_id'])):
        reason = None
        if row['article_title'] in titles or row['article_text_sha256'] in texts: reason = 'historical_article'
        elif row['article_title'] in selected_titles or row['article_text_sha256'] in selected_texts: reason = 'duplicate_article'
        elif len(tokens) < length: reason = 'short_article'
        else:
            # Full article checked against historical windows, not just the selected prefix.
            if spans(tokens) & forbidden: reason = 'historical_64_token_overlap'
            elif spans(tokens) & selected_spans: reason = 'new_article_64_token_overlap'
        if reason:
            rejected.append(dict(article_id=row['article_id'], reason=reason)); continue
        prefix = torch.tensor(tokens[:length], dtype=torch.int64)
        accepted.append((dict(row, article_token_count=len(tokens), token_start=0, token_stop=length,
                              token_hash=mo.digest_tensor(prefix)), prefix))
        selected_titles.add(row['article_title']); selected_texts.add(row['article_text_sha256'])
        selected_spans.update(spans(tokens))
        if len(accepted) == n_search+n_test: break
    return accepted, rejected


def build(root, config, identity, assets):
    folder = root/'data'; manifest = folder/'article_manifest.json'
    if manifest.exists():
        record = read(manifest); require(record['identity'] == identity, 'Data identity changed')
        for role in ('search', 'test'):
            require(sha_file(folder/f'{role}_windows.safetensors') == record['files'][role], 'Data file changed')
        return record
    import datasets
    from transformers import AutoTokenizer
    from safetensors.torch import load_file
    source = Path(config['wikitext'])
    raw = {role:datasets.load_from_disk(str(source/role)) for role in ('train','validation')}
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True, trust_remote_code=False)
    oldmeta = read(Path(config['assets'])/'exp03/data/fit_windows.json')
    titles = {w['article_title'] for w in oldmeta['windows']}
    texts = {w['article_text_sha256'] for w in oldmeta['windows']}
    # Conservatively exclude every validation article: parent windows were concatenated there.
    for _, _, title, text in articles(raw['validation']):
        titles.add(title); texts.add(hashlib.sha256(text.encode()).hexdigest())
    forbidden = set(); history = []
    for path in assets['historical_windows']:
        data = load_file(path); ids = data['input_ids']
        require(ids.ndim == 2 and ids.shape[1] == PLAN['L'], 'Historical window length mismatch')
        if 'attention_mask' in data: require(bool(data['attention_mask'].all()), 'Historical padding is unsupported')
        for row in ids.tolist(): forbidden.update(spans(row))
        history.append(dict(path=path, sha256=sha_file(path), count=len(ids), token_hashes=[mo.digest_tensor(v) for v in ids]))
    for path in config.get('extra_article_manifests', []):
        record = read(path)
        for row in record['windows']:
            titles.add(row['article_title']); texts.add(row['article_text_sha256'])
    candidates = []
    for start, stop, title, text in articles(raw['train']):
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        row = dict(split='train', source_row_start=start, source_row_stop=stop, article_title=title,
                   article_text_sha256=text_hash, article_id=digest(['wikitext-train', title, text_hash]))
        tokens = tokenizer(text, add_special_tokens=False, return_attention_mask=False)['input_ids']
        candidates.append((row, tokens))
    accepted, rejected = select_articles(candidates, forbidden, titles, texts,
        PLAN['search_windows'], PLAN['test_windows'], PLAN['L'])
    required = PLAN['search_windows']+PLAN['test_windows']
    if len(accepted) != required:
        save_json(folder/'data_shortfall.json', dict(required=required, available=len(accepted), rejected=rejected))
        raise RuntimeError(f'Only {len(accepted)}/{required} unseen eligible articles; no reuse/concatenation allowed')
    rows = []; files = {}
    for role, part in [('search', accepted[:PLAN['search_windows']]), ('test', accepted[PLAN['search_windows']:])]:
        windows = [dict(r, role=role, window=i) for i, (r, _) in enumerate(part)]
        tensors = torch.stack([t for _, t in part]); masks = torch.ones_like(tensors)
        save_tensors(folder/f'{role}_windows.safetensors', dict(input_ids=tensors, attention_mask=masks),
                     dict(identity=identity, role=role, windows=windows, mask_hash=mo.digest_tensor(masks)))
        rows.extend(windows); files[role] = sha_file(folder/f'{role}_windows.safetensors')
    record = dict(identity=identity, windows=rows, files=files, history=history, rejected=rejected,
        dataset_files=assets['dataset_files'], tokenizer_files=assets['tokenizer_files'],
        selection='namespace SHA256 order; one 2048-token article prefix; search first32, test next16',
        excluded_title_count=len(titles), excluded_text_count=len(texts),
        overlap_check='exact contiguous 64-token bytes; historical windows and full selected articles; title/text identity',
        official_test_split_used=False, shared_64_token_spans=0, add_special_tokens=False,
        historical_lineage=assets['historical_lineage'])
    save_json(manifest, record)
    return record
