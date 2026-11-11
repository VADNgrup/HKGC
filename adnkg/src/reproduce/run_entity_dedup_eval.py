import argparse
import csv
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

import networkx as nx
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

load_dotenv(SRC_ROOT / ".env", override=False)

from minirag import MiniRAG
from minirag.fuzzy_alignment import (
    MIN_EVENT_DESCRIPTION_COVERAGE,
    MIN_EVENT_DESCRIPTION_SHARED_TOKENS,
)
from minirag.llm import ollama_embed, ollama_model_complete
from minirag.utils import EmbeddingFunc


DEFAULT_SOURCE = REPO_ROOT / "dataset" / ".raw" / "entities.csv"
DEFAULT_GOLD = REPO_ROOT / "dataset" / "entities_gold.json"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "entity-dedup"
GRAPH_FILENAME = "graph_chunk_entity_relation.graphml"
GRAPH_FIELD_SEP = "<SEP>"


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return f"external file (not bundled): {path.name}"


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = value.strip().strip("\"'“”‘’")
    return " ".join(value.split())


def article_id_from_url(value: str) -> str:
    slug = unquote(urlsplit(value).path.rsplit("/", 1)[-1], errors="replace")
    slug = unicodedata.normalize("NFKD", slug)
    slug = "".join(char for char in slug if not unicodedata.combining(char))
    slug = re.sub(r"[^a-zA-Z0-9]", "", slug).casefold()
    if slug.startswith("binhdanhocv"):
        return "binh_dan_hoc_vu"
    if slug.startswith("caotraokhangnhatcuunu"):
        return "cao_trao_khang_nhat_cuu_nuoc"
    return slug


def read_documents(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8-sig", newline="") as source_file:
        rows = list(csv.DictReader(source_file))

    documents: dict[str, dict] = {}
    for row_number, row in enumerate(rows, start=2):
        paragraph = " ".join((row.get("Paragraph") or "").split())
        if not paragraph:
            continue
        url = (row.get("ref_url") or "").strip()
        article_id = article_id_from_url(url)
        if not article_id:
            raise ValueError(f"Missing article URL for non-empty paragraph at row {row_number}")
        document = documents.setdefault(
            article_id,
            {
                "article_id": article_id,
                "paragraphs": [],
                "source_urls": [],
            },
        )
        document["paragraphs"].append(paragraph)
        if url and url not in document["source_urls"]:
            document["source_urls"].append(url)

    if not documents:
        raise ValueError(f"No non-empty Paragraph records found in {csv_path}")
    return [
        {
            "article_id": document["article_id"],
            "text": "\n\n".join(document["paragraphs"]),
            "paragraph_count": len(document["paragraphs"]),
            "source_urls": document["source_urls"],
        }
        for document in documents.values()
    ]


def parse_aliases(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, list):
        return [str(item) for item in decoded if str(item).strip()]
    return [
        part.strip()
        for part in re.split(r",\s*|" + re.escape(GRAPH_FIELD_SEP), value)
        if part.strip()
    ]


def load_gold_surfaces(
    gold_path: Path,
) -> tuple[dict[str, str], set[str], dict[str, str], list[dict]]:
    gold_data = json.loads(gold_path.read_text(encoding="utf-8"))
    surface_to_entity: dict[str, str] = {}
    surface_to_display: dict[str, str] = {}
    gold_entities: set[str] = set()
    gold_mentions = []
    for entity in gold_data["entities"]:
        entity_id = entity["id"]
        gold_entities.add(entity_id)
        for surface in [entity["name"], *entity.get("alias", [])]:
            normalized = normalize_name(surface)
            previous = surface_to_entity.get(normalized)
            if previous is not None and previous != entity_id:
                raise ValueError(
                    f"Gold surface {surface!r} belongs to both {previous} and {entity_id}"
                )
            surface_to_entity[normalized] = entity_id
            surface_to_display.setdefault(normalized, surface)
        gold_mentions.extend(
            {"gold_entity_id": entity_id, **mention}
            for mention in entity.get("mentions", [])
        )
    return surface_to_entity, gold_entities, surface_to_display, gold_mentions


def evaluate_graph(graph_path: Path, gold_path: Path) -> dict:
    if not graph_path.is_file():
        raise FileNotFoundError(f"Indexing did not produce graph file: {graph_path}")
    graph = nx.read_graphml(graph_path)
    (
        surface_to_entity,
        gold_entities,
        surface_to_display,
        gold_mentions,
    ) = load_gold_surfaces(gold_path)

    surface_to_nodes: dict[str, set[str]] = defaultdict(set)
    node_gold_surfaces: dict[str, set[str]] = defaultdict(set)
    for node_id, attributes in graph.nodes(data=True):
        node_key = str(node_id)
        surfaces = [node_key, *parse_aliases(attributes.get("alternate_names"))]
        for surface in surfaces:
            normalized = normalize_name(surface)
            if normalized in surface_to_entity:
                surface_to_nodes[normalized].add(node_key)
                node_gold_surfaces[node_key].add(normalized)

    represented = {
        surface: next(iter(node_ids))
        for surface, node_ids in surface_to_nodes.items()
        if len(node_ids) == 1
    }
    ambiguous_surfaces = sorted(
        surface for surface, node_ids in surface_to_nodes.items() if len(node_ids) > 1
    )
    represented_mentions = []
    for mention in gold_mentions:
        normalized = normalize_name(mention["surface"])
        node_ids = surface_to_nodes.get(normalized, set())
        if len(node_ids) == 1:
            represented_mentions.append(
                {
                    "mention_id": mention["id"],
                    "gold_entity_id": mention["gold_entity_id"],
                    "surface": mention["surface"],
                    "predicted_node": next(iter(node_ids)),
                }
            )
    tp = fp = fn = 0
    false_merge_pairs = []
    missed_merge_pairs = []
    for left_index, left in enumerate(represented_mentions):
        for right in represented_mentions[left_index + 1 :]:
            left_gold_id = left["gold_entity_id"]
            right_gold_id = right["gold_entity_id"]
            gold_same = left_gold_id == right_gold_id
            predicted_same = left["predicted_node"] == right["predicted_node"]
            if gold_same and predicted_same:
                tp += 1
            elif not gold_same and predicted_same:
                fp += 1
                false_merge_pairs.append(
                    {
                        "left_mention_id": left["mention_id"],
                        "left_surface": left["surface"],
                        "left_gold_entity_id": left_gold_id,
                        "right_mention_id": right["mention_id"],
                        "right_surface": right["surface"],
                        "right_gold_entity_id": right_gold_id,
                        "predicted_node": left["predicted_node"],
                    }
                )
            elif gold_same and not predicted_same:
                fn += 1
                missed_merge_pairs.append(
                    {
                        "left_mention_id": left["mention_id"],
                        "left_surface": left["surface"],
                        "right_mention_id": right["mention_id"],
                        "right_surface": right["surface"],
                        "gold_entity_id": left_gold_id,
                        "left_predicted_node": left["predicted_node"],
                        "right_predicted_node": right["predicted_node"],
                    }
                )

    predicted_positive = tp + fp
    gold_positive = tp + fn
    precision = (
        tp / predicted_positive
        if predicted_positive
        else (0.0 if gold_positive else None)
    )
    recall = tp / gold_positive if gold_positive else None
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else None
    represented_gold_entities = {
        mention["gold_entity_id"] for mention in represented_mentions
    }
    matched_node_ids = {
        node_id for node_id in graph.nodes if str(node_id) in node_gold_surfaces
    }
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "gold_entities_total": len(gold_entities),
        "gold_entities_represented": len(represented_gold_entities),
        "gold_surface_forms_represented": len(represented),
        "gold_mentions_represented": len(represented_mentions),
        "gold_surface_forms_ambiguous": len(ambiguous_surfaces),
        "ambiguous_surface_forms": [
            surface_to_display[surface] for surface in ambiguous_surfaces
        ],
        "evaluated_mention_pairs": (
            len(represented_mentions) * (len(represented_mentions) - 1) // 2
        ),
        "gold_positive_pairs_evaluated": gold_positive,
        "predicted_positive_pairs": predicted_positive,
        "tp": tp,
        "fp_false_merge": fp,
        "fn_missed_merge": fn,
        "false_merge_pairs": false_merge_pairs,
        "missed_merge_pairs": missed_merge_pairs,
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "graph_nodes_with_gold_match": len(matched_node_ids),
        "graph_nodes_outside_gold_inventory": graph.number_of_nodes() - len(matched_node_ids),
    }


def export_entity_list(graph_path: Path, output_dir: Path) -> dict:
    graph = nx.read_graphml(graph_path)
    entities = []
    for node_id, attributes in graph.nodes(data=True):
        entities.append(
            {
                "entity_id": str(node_id),
                "entity_type": attributes.get("entity_type", ""),
                "aliases": parse_aliases(attributes.get("alternate_names")),
                "description": attributes.get("description", ""),
                "source_id": attributes.get("source_id", ""),
                "time_range": attributes.get("time_range", ""),
                "geo_location": attributes.get("geo_location", ""),
                "degree": graph.degree[node_id],
            }
        )
    entities.sort(key=lambda entity: normalize_name(entity["entity_id"]))

    json_path = output_dir / "entities.json"
    json_path.write_text(
        json.dumps(entities, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = output_dir / "entities.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=[
                "entity_id",
                "entity_type",
                "aliases",
                "description",
                "source_id",
                "time_range",
                "geo_location",
                "degree",
            ],
        )
        writer.writeheader()
        for entity in entities:
            row = dict(entity)
            row["aliases"] = json.dumps(row["aliases"], ensure_ascii=False)
            writer.writerow(row)
    return {
        "entity_count": len(entities),
        "json_path": json_path.name,
        "csv_path": csv_path.name,
    }


def create_rag(
    args, work_dir: Path, enable_fuzzy_logic: bool
) -> MiniRAG:
    embedding_func = EmbeddingFunc(
        embedding_dim=args.embedding_dim,
        max_token_size=1000,
        func=lambda texts: ollama_embed(
            texts,
            embed_model=args.embedding_model,
            host=args.llm_host,
            timeout=args.timeout,
        ),
    )
    return MiniRAG(
        working_dir=str(work_dir),
        graph_storage="NetworkXStorage",
        llm_model_func=ollama_model_complete,
        llm_model_name=args.llm_model,
        llm_model_kwargs={
            "host": args.llm_host,
            "timeout": args.timeout,
            "options": {"temperature": 0},
        },
        embedding_func=embedding_func,
        enable_fuzzy_logic=enable_fuzzy_logic,
        fl_min_shared_neighbors=args.min_shared_neighbors,
    )


def format_metric(value) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_results(run_dir: Path, results: dict, metadata: dict) -> None:
    payload = {"metadata": metadata, "methods": results}
    (run_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = [
        "| Method | Nodes | Edges | Gold entities represented | Gold mentions | Gold positive pairs | Pair P | Pair R | Pair F1 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    labels = {
        "minirag": "MiniRAG",
        "ndakg": f"NDAKG (CRA, shared-neighbor min={metadata['cra_min_shared_neighbors']})",
    }
    for method, result in results.items():
        rows.append(
            "| {method} | {nodes} | {edges} | {matched}/{total} | {mentions} | "
            "{gold_pairs} | {precision} | {recall} | {f1} | {tp} | {fp} | {fn} |".format(
                method=labels[method],
                nodes=result["nodes"],
                edges=result["edges"],
                matched=result["gold_entities_represented"],
                total=result["gold_entities_total"],
                mentions=result["gold_mentions_represented"],
                gold_pairs=result["gold_positive_pairs_evaluated"],
                precision=format_metric(result["pairwise_precision"]),
                recall=format_metric(result["pairwise_recall"]),
                f1=format_metric(result["pairwise_f1"]),
                tp=result["tp"],
                fp=result["fp_false_merge"],
                fn=result["fn_missed_merge"],
            )
        )
    rows.append(
        "\nPair metrics are conditional on gold mention occurrences whose surface is represented in each indexed graph. "
        "Unmatched graph nodes are reported separately and are not treated as false positives."
    )
    (run_dir / "table.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    latex_rows = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Direct entity deduplication evaluation on the consensus-tuple subset.}",
        r"\label{tab:entity-dedup-direct}",
        r"\begin{tabular}{l|rrrrr|rrr|rrr}",
        r"\toprule",
        r"Method & Nodes & Edges & Gold coverage & Gold mentions & Gold positive pairs & Precision & Recall & F1 & TP & FP & FN \\",
        r"\midrule",
    ]
    for method, result in results.items():
        latex_rows.append(
            "{} & {} & {} & {}/{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
                labels[method],
                result["nodes"],
                result["edges"],
                result["gold_entities_represented"],
                result["gold_entities_total"],
                result["gold_mentions_represented"],
                result["gold_positive_pairs_evaluated"],
                format_metric(result["pairwise_precision"]),
                format_metric(result["pairwise_recall"]),
                format_metric(result["pairwise_f1"]),
                result["tp"],
                result["fp_false_merge"],
                result["fn_missed_merge"],
            )
        )
    latex_rows.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (run_dir / "table.tex").write_text("\n".join(latex_rows) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Index the entity-evaluation paragraphs with MiniRAG and NDAKG/CRA, "
            "then compare graph nodes against the curated gold entity inventory."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--gold-entities", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--run-id",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Unique subdirectory name; existing runs are never overwritten.",
    )
    parser.add_argument(
        "--llm-host",
        default=os.getenv("LLM_HOST", "http://localhost:11434"),
    )
    parser.add_argument("--llm-model", default=os.getenv("LLM_MODEL", "qwen3.5:9b"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBED_MODEL"))
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=int(os.getenv("EMBED_MODEL_DIM", "4096")),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        help="CRA identity/neighborhood threshold used for this evaluation.",
    )
    parser.add_argument(
        "--min-shared-neighbors",
        type=int,
        default=2,
        help=(
            "Minimum number of shared graph neighbors required for a "
            "neighborhood-based merge; use 1 to reproduce the published rule."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("REQUEST_TIMEOUT", "600")),
    )
    args = parser.parse_args()

    if not args.embedding_model:
        parser.error(
            "Set EMBED_MODEL in adnkg/.env or pass --embedding-model; "
            "the Ollama embedding model must be available at --llm-host."
        )
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1.")
    if args.min_shared_neighbors < 1:
        parser.error("--min-shared-neighbors must be at least 1.")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        parser.error("--run-id may contain only letters, digits, dot, underscore, and hyphen.")
    if not args.source.is_file():
        parser.error(f"Source CSV does not exist: {args.source}")
    if not args.gold_entities.is_file():
        parser.error(
            f"Gold entity inventory does not exist: {args.gold_entities}. "
            "Run prepare_entity_dedup_gold.py first."
        )

    documents = read_documents(args.source)
    load_gold_surfaces(args.gold_entities)
    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    os.environ["FL_THRES_SIM"] = str(args.threshold)
    os.environ["FL_THRES_MAP"] = str(args.threshold)
    metadata = {
        "source_csv": portable_path(args.source),
        "gold_entities": portable_path(args.gold_entities),
        "article_count": len(documents),
        "paragraph_count": sum(doc["paragraph_count"] for doc in documents),
        "articles": [
            {
                "article_id": doc["article_id"],
                "paragraph_count": doc["paragraph_count"],
                "source_urls": doc["source_urls"],
            }
            for doc in documents
        ],
        "llm_endpoint": "Omitted from portable results; configure LLM_HOST locally.",
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "embedding_dim": args.embedding_dim,
        "temperature": 0,
        "cra_threshold": args.threshold,
        "cra_min_shared_neighbors": args.min_shared_neighbors,
        "cra_event_description_min_shared_tokens": (
            MIN_EVENT_DESCRIPTION_SHARED_TOKENS
        ),
        "cra_event_description_min_coverage": MIN_EVENT_DESCRIPTION_COVERAGE,
        "same_prompt_and_indexing_settings": True,
    }
    results = {}

    for method, enable_fuzzy_logic in (("minirag", False), ("ndakg", True)):
        work_dir = run_dir / method
        work_dir.mkdir()
        print(
            f"\n[{method}] Indexing {len(documents)} article documents "
            f"({metadata['paragraph_count']} source paragraphs)..."
        )
        rag = create_rag(args, work_dir, enable_fuzzy_logic)
        rag.insert([doc["text"] for doc in documents])

        graph_path = work_dir / GRAPH_FILENAME
        results[method] = evaluate_graph(graph_path, args.gold_entities)
        results[method]["entity_list"] = export_entity_list(graph_path, work_dir)
        write_results(run_dir, results, metadata)
        print(
            f"[{method}] nodes={results[method]['nodes']}, "
            f"pairwise F1={format_metric(results[method]['pairwise_f1'])}"
        )

    write_results(run_dir, results, metadata)
    print(f"\nSaved direct deduplication comparison to {run_dir}")
    print((run_dir / "table.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
