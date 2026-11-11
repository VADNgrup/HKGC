# HKGC

This repository contains code and data for historical knowledge-graph construction research. The NDAKG implementation, paper, and direct entity-deduplication experiment are in [`adnkg/`](./adnkg/readme.md). Dataset preparation, annotation provenance, and reproduction instructions are documented in [`adnkg/dataset/readme.md`](./adnkg/dataset/readme.md).

## Reproduce the direct entity-deduplication experiment

The experiment compares MiniRAG's native indexing pipeline with NDAKG/CRA on two article-level documents assembled from the annotated paragraphs. It produces each method's graph and entity list, plus pairwise precision, recall, and F1 against the curated gold inventory.

From the repository root, in a Bash shell:

```bash
cd adnkg
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-entity-dedup.txt
cp -n src/.env.example src/.env
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:latest
bash ./src/scripts/step_x.sh
```

The Ollama service must be reachable at `LLM_HOST` in `src/.env` (the example uses localhost). The embedding model dimension is set to 4096 in the example. Override the defaults with `LLM_HOST`, `LLM_MODEL`, `EMBED_MODEL`, `EMBED_MODEL_DIM`, `THRESHOLD`, `MIN_SHARED_NEIGHBORS`, `RUN_ID`, or `OUTPUT_DIR`.

The committed compact result bundle is under `adnkg/output/entity-dedup/direct-dedup-release-20261111/`. New runs are written to `adnkg/output/entity-dedup/<run-id>/`; large vector stores and caches are intentionally excluded from version control.

## Reproduce Q&A judging

`adnkg/src/compair_ans.py` evaluates generated answer CSV files with an OpenAI-compatible judge endpoint. Configure `QNA_EVAL_ENDPOINT`, `QNA_EVAL_API_KEY`, and `QNA_EVAL_MODEL` in the local `adnkg/src/.env`; the example values target a local Ollama server. The published Q&A scores are single-run descriptive results, not statistical significance claims.

For the dataset and gold-entity inventory details, see [`adnkg/dataset/readme.md`](./adnkg/dataset/readme.md).
