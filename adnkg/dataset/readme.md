# NDAKG dataset and entity-deduplication subset

This directory contains the historical QA benchmark preparation scripts and the small manually annotated dataset used for direct entity-deduplication evaluation. The direct-evaluation data are tracked in [`.raw/entities.csv`](./.raw/entities.csv), and the generated canonical inventory is [`entities_gold.json`](./entities_gold.json).

## Direct-deduplication data

`.raw/entities.csv` contains 29 paragraph rows from two Vietnamese Wikipedia articles, with a reference URL and consensus gold tuples for each paragraph. Each tuple contributes subject and object mentions to the gold inventory. Paragraphs are useful annotation units, but the indexing experiment groups rows by article URL, preserves their CSV order, and concatenates them into two article-level documents before indexing. It does not index each paragraph as a separate document.

The gold file contains 187 canonical entities, their aliases, mention occurrences, source rows, article IDs, and a `manual_merges` decision log with rationales. The manual aliases resolve spelling/punctuation variants and contextual labels reviewed for these articles, including the Japanese wartime actor labels and the two names for the Bình dân học vụ movement. These are dataset-specific annotation decisions, not a general-purpose alias rule.

Regenerate the gold inventory from the checked-in annotations (no model/API required):

```bash
cd adnkg
python ./src/reproduce/prepare_entity_dedup_gold.py
```

To keep generated output elsewhere:

```bash
python ./src/reproduce/prepare_entity_dedup_gold.py \
  --output /tmp/entities_gold.json
```

The generation is deterministic. Review changes to `manual_merges` in `src/reproduce/prepare_entity_dedup_gold.py` as annotation changes, not model predictions. The runner validates that each normalized gold surface maps to only one canonical entity.

## Reproduce indexing and pair evaluation

The direct evaluation uses the indexed graph's node names and alternate names to map annotated surfaces to predicted nodes. It reports pairwise precision, recall, and F1 over represented gold mention occurrences, along with TP, false-merge FP, missed-merge FN, and gold entity/mention coverage. Unmatched graph nodes are not counted as pairwise false positives, and ambiguous gold surfaces are excluded. Therefore, pairwise scores are conditional on coverage and must not be reported alone.

Follow the environment and indexing instructions in [`../readme.md`](../readme.md), then run `bash ./src/scripts/step_x.sh`. It indexes MiniRAG and NDAKG separately using the same two article-level source documents, exports both KGs and entity lists, and writes the comparison under `adnkg/output/entity-dedup/<run-id>/`. The direct experiment does not run the QA benchmark.

## Main benchmark preparation

The existing scripts `0.pull_data_from_wiki.py` and `1.process_qna_sheet.py` prepare the larger historical corpus and question-answer set. See the top-level [`../readme.md`](../readme.md) for the repository workflow and [`../docs/paper.md`](../docs/paper.md) for the benchmark description and limitations.

## Source attribution

Paragraph text is derived from the Wikipedia pages referenced in the CSV. Preserve those source URLs and comply with the applicable Wikipedia/Wikimedia attribution and share-alike license terms when redistributing or adapting the text. The gold tuples, canonical entity assignments, and alias rationales are annotation artifacts maintained for this experiment.
