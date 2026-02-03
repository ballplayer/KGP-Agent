import os
import json
import math
import numpy as np
try:
    import google.generativeai as genai
except Exception:
    genai = None
import networkx as nx

# Configurable weights
ALPHA = 0.7  # embedding similarity weight
BETA = 0.2   # keyword / alias match weight
GAMMA = 0.1  # graph centrality weight

EMBED_MODEL = "models/text-embedding-004"


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_text(s: str) -> str:
    if not s:
        return ""
    return ''.join(ch for ch in s).strip()


def cosine(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def ensure_kb_embeddings(kb_path):
    """Ensure KB topics have embeddings; update file if new embeddings were computed."""
    if genai is None:
        return

    kb = load_json(kb_path)
    texts = []
    entries = []
    for subj, sdata in kb.get('subjects', {}).items():
        for topic in sdata.get('topics', []):
            if not topic.get('embedding'):
                texts.append(topic.get('title') + ' ' + ' '.join(topic.get('keywords', [])) + ' ' + topic.get('description', ''))
                entries.append((subj, topic))

    if not texts:
        return

    # call embedding API in batch
    try:
        resp = genai.embed_content(model=EMBED_MODEL, content=texts, task_type="semantic_similarity")
        # genai.embed_content may return dict with 'embedding' or list; try to handle common shapes
        embeddings = None
        if isinstance(resp, dict) and 'embedding' in resp:
            embeddings = resp['embedding']
        else:
            # Some clients return a list of embeddings in resp
            embeddings = resp

        # Assign back
        for (subj, topic), emb in zip(entries, embeddings):
            topic['embedding'] = emb

        # Persist
        save_json(kb, kb_path)
        print(f"✅ 已为 {len(entries)} 个知识点生成并缓存 embedding 到 {kb_path}")
    except Exception as e:
        print(f"⚠️ 生成 KB embedding 失败: {e}")


def embed_texts(texts):
    if genai is None:
        return [None] * len(texts)
    try:
        resp = genai.embed_content(model=EMBED_MODEL, content=texts, task_type="semantic_similarity")
        if isinstance(resp, dict) and 'embedding' in resp:
            return resp['embedding']
        return resp
    except Exception as e:
        print(f"⚠️ embed_texts 调用失败: {e}")
        return [None] * len(texts)


def extract_candidates_from_kg(kg_data):
    candidates = []
    ent_map = {e['id']: e for e in kg_data.get('entities', [])}

    # entity ids and types
    for e in kg_data.get('entities', []):
        candidates.append({'text': e.get('id', ''), 'entity_id': e.get('id'), 'type': 'id'})
        # attributes
        for k, v in (e.get('attributes') or {}).items():
            candidates.append({'text': f"{k}: {v}", 'entity_id': e.get('id'), 'type': 'attribute'})

    # relations
    for r in kg_data.get('relationships', []):
        txt = f"{r.get('source')} {r.get('relation')} {r.get('target')}"
        candidates.append({'text': txt, 'entity_id': r.get('source'), 'type': 'relation'})

    # full summary
    candidates.append({'text': json.dumps(kg_data, ensure_ascii=False), 'entity_id': None, 'type': 'full'})

    # dedupe by text
    seen = set()
    unique = []
    for c in candidates:
        t = c['text']
        if t in seen:
            continue
        seen.add(t)
        unique.append(c)
    return unique


def compute_centrality_scores(kg_data):
    G = nx.Graph()
    for e in kg_data.get('entities', []):
        G.add_node(e['id'])
    for r in kg_data.get('relationships', []):
        G.add_edge(r['source'], r['target'])
    if len(G) == 0:
        return {}
    deg = nx.degree_centrality(G)  # normalized
    return deg


def match_and_rank(kg_path, kb_path, output_dir):
    kg = load_json(kg_path)
    kb = load_json(kb_path)

    # ensure KB embeddings exist (will update file)
    ensure_kb_embeddings(kb_path)

    candidates = extract_candidates_from_kg(kg)
    texts = [normalize_text(c['text']) for c in candidates]

    # generate embeddings for candidates
    cand_embs = embed_texts(texts)

    # collect KB embeddings into a flat list with metadata
    kb_entries = []  # dicts: {subj, topic}
    for subj, sdata in kb.get('subjects', {}).items():
        for topic in sdata.get('topics', []):
            kb_entries.append({'subject': subj, 'topic': topic})

    kb_texts = [t['topic'].get('title', '') + ' ' + ' '.join(t['topic'].get('keywords', [])) for t in kb_entries]
    kb_embs = [t['topic'].get('embedding') for t in kb_entries]

    # fallback: if no embeddings for KB / candidates, do keyword matching only
    use_embedding = all(e is not None for e in kb_embs) and any(e is not None for e in cand_embs)

    centrality = compute_centrality_scores(kg)

    scores = []
    for i, entry in enumerate(kb_entries):
        topic = entry['topic']
        max_sim = 0.0
        best_evidence = None

        # embedding similarity
        if use_embedding:
            kb_vec = np.array(kb_embs[i], dtype=float)
            for j, c_emb in enumerate(cand_embs):
                if c_emb is None:
                    continue
                sim = cosine(kb_vec, np.array(c_emb, dtype=float))
                if sim > max_sim:
                    max_sim = sim
                    best_evidence = {'candidate': candidates[j]['text'], 'sim': sim, 'candidate_type': candidates[j]['type']}

        # keyword / alias match score (simple)
        keyword_score = 0.0
        kw_list = topic.get('keywords', []) + topic.get('aliases', [])
        kw_lower = [k.lower() for k in kw_list]
        for c in candidates:
            txt = c['text'].lower()
            for kw in kw_lower:
                if kw and kw in txt:
                    keyword_score = max(keyword_score, 1.0)  # binary for now

        # centrality score: max centrality of any matched entity for this topic
        cent_score = 0.0
        for c in candidates:
            ent_id = c.get('entity_id')
            if ent_id and ent_id in centrality:
                # if any keyword matches this candidate, boost its cent score
                cent_score = max(cent_score, centrality.get(ent_id, 0.0))

        # combine
        emb_sim = max_sim if max_sim is not None else 0.0
        combined = ALPHA * emb_sim + BETA * keyword_score + GAMMA * cent_score

        scores.append({
            'subject': entry['subject'],
            'id': topic.get('id'),
            'title': topic.get('title'),
            'emb_sim': emb_sim,
            'keyword_score': keyword_score,
            'centrality': cent_score,
            'combined_score': combined,
            'evidence': best_evidence
        })

    # sort
    scores_sorted = sorted(scores, key=lambda x: x['combined_score'], reverse=True)

    # pick top 2 with threshold logic
    top1 = scores_sorted[0] if scores_sorted else None
    top2 = scores_sorted[1] if len(scores_sorted) > 1 else None

    threshold_high = 0.80
    threshold_mid = 0.65
    margin = 0.15

    selected = []
    if top1:
        if top1['combined_score'] >= threshold_high:
            selected = [top1]
        elif top2 and top1['combined_score'] >= threshold_mid and (top1['combined_score'] - top2['combined_score'] >= margin):
            selected = [top1]
        else:
            selected = [top1] + ([top2] if top2 else [])

    result = {
        'candidates': scores_sorted[:10],
        'selected': selected
    }

    # save
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, '6_core_concepts.json')
    save_json(result, out_path)
    print(f"✅ 核心知识点结果已保存: {out_path}")
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--kg', default='KG_P.json')
    parser.add_argument('--kb', default='exam_kb.json')
    parser.add_argument('--out', default='.')
    args = parser.parse_args()
    match_and_rank(args.kg, args.kb, args.out)
