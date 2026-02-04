# Multi-Modal Knowledge Graph → Exam Concept Pipeline

This repository provides a compact pipeline that converts images (plus optional text) into a structured knowledge graph, prunes it with a hybrid approach, analyzes and visualizes results, and finally maps the pruned graph to likely exam knowledge points.

---

## At a glance

- Input: image (e.g., `test.png`) + optional descriptive text
- Pipeline: image summary → KG extraction → hybrid pruning → analysis → SVG generation → KB matching
- Output: timestamped `Output_YYYYMMDD_HHMMSS` folder with JSONs, images, Excel, and `6_core_concepts.json`

Recent updates:
- `exam_kb.json` expanded to 500 template topics (physics/math/chemistry)
- `match_kb.py` added and integrated into `workflow.py` (per-run KB matching)
- KB embeddings are cached in `exam_kb.json` (generated once per topic)

---

## Quickstart (PowerShell)

1. Set environment variables for the session (API key and optional proxy):

```powershell
$env:GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
$env:HTTP_PROXY = 'http://127.0.0.1:7890'   # optional
$env:HTTPS_PROXY = 'http://127.0.0.1:7890'  # optional
```

2. Install dependencies if needed:

```powershell
pip install -r requirements.txt
```

3. Run the pipeline (interactive local mode saves outputs and opens viewer windows):

```powershell
python .\workflow.py
```

Tip: the pipeline will create a folder like `Output_20260203_115747` containing all artifacts.

---

## Files & Useful commands

- `exam_kb.json` — exam knowledge base (500 topics). You can regenerate it with `tools/generate_exam_kb.py`.
- `match_kb.py` — KB matching module (embedding + scoring + caching). Run standalone:

```powershell
python .\match_kb.py --kg Output_XXXX\3_KG_P.json --kb exam_kb.json --out Output_XXXX
```

- `tools/generate_exam_kb.py` — regenerate the template KB with configurable counts per subject.

---

## match_kb: what to expect

- Extracts candidate texts from the pruned KG (entity ids, attributes, relations and a compact KG summary).
- Uses embeddings + simple keyword/alias matching + graph centrality to compute a combined score for each KB topic.
- Writes `6_core_concepts.json` to the output folder with candidates and selected top 1–2 concepts (plus evidence).

If you change KB text or the embedding model, re-run the KB embedding step — `match_kb` will auto-generate and cache missing embeddings on first run.

---

## Embedding & cost note

- Generating embeddings for 500 topics was performed by the pipeline and cached into `exam_kb.json`. That step consumes API quota; subsequent runs reuse cached vectors.
- To refresh embeddings: remove the `embedding` fields or run the ensure/update helper in `match_kb.py`.

---

## Troubleshooting & tips

- `403` or key errors: confirm `$env:GOOGLE_API_KEY` and proxy settings.
- FutureWarning: the project currently uses `google.generativeai` (deprecated). Plan migration to `google.genai`.
- Lint noise: add `.flake8` to exclude environment directories (example in the repo).

---

## 📄 License


This project is provided for educational and research purposes.

