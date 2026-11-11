# NDAKG: Historical Knowledge-Graph Construction

This folder contains the NDAKG implementation, the MiniRAG-based baseline, experiment scripts, and the paper source. NDAKG adds a historical entity schema and Context-Aware Rule-Based Node Alignment (CRA) to MiniRAG's graph-indexing pipeline. The implementation and experiment described here are in [`src/`](./src/); the paper source is [`docs/paper.md`](./docs/paper.md).

## CRA overview

CRA preserves the paper's two-phase alignment structure:

1. **Within a chunk:** Union-Find clusters candidate nodes when identity-token similarity exceeds the configured threshold. If both clusters have known entity types, the types must agree; unknown types cannot bridge incompatible known types.
2. **Against the core graph:** candidates of incompatible known types are rejected, then CRA evaluates identity similarity, neighborhood similarity, and temporal status. The direct-dedup configuration uses an identity/neighborhood threshold of 0.80 and requires at least two shared neighbors for neighborhood-based merges. Non-identical organization names also need that context support.

Known, non-overlapping time ranges block ordinary neighborhood and identity merges. The narrow exception is for two EVENT nodes: they must meet the identity threshold and their descriptions must share at least three content tokens and at least 50% of the smaller content-token set. Otherwise, a high-identity pair with a temporal mismatch remains distinct and receives a directed `RENAMED_TO` temporal mapping. The exception adds evidence to the existing temporal rule; it does not change the overall two-phase algorithm.

## Direct entity-deduplication evaluation

The labeled subset is described in [`dataset/readme.md`](./dataset/readme.md). The runner joins paragraphs by article URL, concatenates each article's paragraphs in source order, and indexes one document per article. MiniRAG and NDAKG index the same article-level source documents and settings independently; the comparison is end-to-end, not a post-extraction alignment test on an identical candidate list.

### Setup and run

From this directory, in Bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-entity-dedup.txt
cp -n src/.env.example src/.env
ollama pull qwen3.5:9b
ollama pull qwen3-embedding:latest
bash ./src/scripts/step_x.sh
```

Set `LLM_HOST` in `src/.env` to the Ollama host. The example uses `http://localhost:11434`, Qwen3.5 9B, and a 4096-dimensional `qwen3-embedding` model. Set `RUN_ID` to preserve a distinct run directory; existing directories are never overwritten.

### Outputs

Each run writes:

- `minirag/` and `ndakg/`: GraphML KGs and `entities.json` / `entities.csv` entity lists.
- `table.md`, `table.tex`, and `results.json`: node/edge counts, gold coverage, and pairwise precision, recall, F1, TP, FP, and FN.
- `ndakg/fuzzy_alignment_decision_log.txt`: CRA merge and temporal-mapping decisions.

Pairwise scores are conditional on annotated mention surfaces represented unambiguously in a graph. Always report mention/entity coverage alongside these scores. The compact public snapshot (graphs, entity lists, logs, and tables; no vector stores) is in [`output/entity-dedup/direct-dedup-release-20261111/`](./output/entity-dedup/direct-dedup-release-20261111/).

## Other scripts and data

- `src/scripts/step_0.sh`: build the benchmark KG.
- `src/scripts/step_1.sh`: test Q&A as in MiniRAG.
- `src/scripts/eval_qna.sh`: evaluate a KG on the benchmark.
- `src/scripts/visualize_kg.sh`: visualize a KG.
- [`dataset/readme.md`](./dataset/readme.md): source data, annotation format, and gold-inventory regeneration.

The direct-dedup experiment does not run Q&A. See the paper's limitations section for the small evaluation size, calibration limitations, and the need for a shared-candidate post-extraction baseline and repeated Q&A measurements.
