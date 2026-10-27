import asyncio
import json
import logging
import os
import re
from contextlib import contextmanager
from typing import Union
from collections import Counter, defaultdict
import warnings
import json_repair
import tqdm
import numpy as np
from .schema import clean_entity_info, clean_entity_name, clean_quotes_and_brackets, parse_time_range, is_noise_entity, Person, Place, TimeEntity, Event, Artifact, ObjectEntity, Concept, BaseEntity
from .utils import (
    list_of_list_to_csv,
    truncate_list_by_token_size,
    split_string_by_multi_markers,
    logger,
    locate_json_string_body_from_string,
    process_combine_contexts,
    clean_str,
    edge_vote_path,
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    is_float_regex,
    pack_user_ass_to_openai_messages,
    compute_mdhash_id,
    calculate_similarity,
    cal_path_score_list,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from .exceptions import APITimeoutError, APIConnectionError, RateLimitError


@contextmanager
def _suppress_httpx_logs():
    logger_names = ("httpx", "httpcore")
    previous_levels = {
        name: logging.getLogger(name).level for name in logger_names
    }
    for name in logger_names:
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        yield
    finally:
        for name, level in previous_levels.items():
            logging.getLogger(name).setLevel(level)


def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
    chunk_text: str = None,
):
    if len(record_attributes) < 4 or clean_quotes_and_brackets(record_attributes[0]).lower() != 'entity':
        return None
    
    raw_name = record_attributes[1]
    raw_type = record_attributes[2]
    
    # Handle both 5-tuple legacy format and new 6-tuple format with alternate_names
    if len(record_attributes) >= 6 and clean_quotes_and_brackets(record_attributes[3]).lower() in {"none", ""} or ";" in record_attributes[3] or "," in record_attributes[3] or len(record_attributes) >= 6:
        # Check if attribute 3 looks like alternate_names
        raw_aliases = record_attributes[3]
        raw_desc = record_attributes[4] if len(record_attributes) >= 5 else ""
        raw_time = record_attributes[5] if len(record_attributes) >= 6 else "unknown"
        raw_geo = record_attributes[6] if len(record_attributes) >= 7 else "unknown"
    else:
        raw_aliases = ""
        raw_desc = record_attributes[3]
        raw_time = record_attributes[4] if len(record_attributes) >= 5 else "unknown"
        raw_geo = record_attributes[5] if len(record_attributes) >= 6 else "unknown"
    
    cleaned = clean_entity_info(raw_name, raw_type, raw_desc, raw_aliases, raw_time, raw_geo)
    if cleaned is None:
        return None
        
    name, ent_type, desc, aliases, time_range, geo = cleaned

    # Check: entity name must exist in the original chunk content (case-insensitive, whitespace normalized)
    if chunk_text is not None:
        name_lower = " ".join(name.lower().split())
        chunk_lower = " ".join(chunk_text.lower().split())
        if name_lower not in chunk_lower:
            return None

    # Check: must not contain special characters (only allow letters, numbers, spaces, and Vietnamese diacritics)
    if not re.match(r"^[a-zA-Z0-9\sàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]+$", name):
        return None
    
    return dict(
        entity_name=name,
        entity_type=ent_type,
        alternate_names=aliases,
        description=desc,
        source_id=chunk_key,
        time_range=time_range,
        geo_location=geo,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or clean_quotes_and_brackets(record_attributes[0]).lower() != 'relationship':
        return None
        
    source = clean_entity_name(record_attributes[1]).upper()
    target = clean_entity_name(record_attributes[2]).upper()
    
    # Noise filtering
    if is_noise_entity(source) or is_noise_entity(target):
        return None
        
    edge_description = clean_quotes_and_brackets(record_attributes[3])
    edge_keywords = clean_quotes_and_brackets(record_attributes[4])
    
    # Extract weight
    weight = 1.0
    if len(record_attributes) >= 6:
        w_str = clean_quotes_and_brackets(record_attributes[5])
        if is_float_regex(w_str):
            weight = float(w_str)
            
    # Extract relationship time range if present
    rel_time = "unknown"
    if len(record_attributes) >= 7:
        rel_time = clean_quotes_and_brackets(record_attributes[6])
        
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        keywords=edge_keywords,
        source_id=chunk_key,
        time_range=rel_time,
    )

def _sanitize_for_graphml(data: dict) -> dict:
    """Remove None values and convert non-primitive types (e.g. list) to
    GraphML-compatible strings. GraphML only supports str, int, float, bool."""
    result = {}
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, list):
            # Convert list to comma-separated string
            result[k] = ", ".join(str(item) for item in v) if v else ""
        elif isinstance(v, dict):
            result[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, (str, int, float, bool)):
            result[k] = v
        else:
            result[k] = str(v)
    return result


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []
    already_time_ranges = []
    already_geo_locations = []

    already_alternate_names = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])
        if "time_range" in already_node and already_node["time_range"]:
            already_time_ranges.append(already_node["time_range"])
        if "geo_location" in already_node and already_node["geo_location"]:
            already_geo_locations.append(already_node["geo_location"])
        if "alternate_names" in already_node:
            raw_alts = already_node["alternate_names"]
            if isinstance(raw_alts, list):
                already_alternate_names.extend(raw_alts)
            elif isinstance(raw_alts, str) and raw_alts:
                already_alternate_names.extend([a.strip() for a in raw_alts.split(",") if a.strip()])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]

    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )

    # Merge alternate names across extractions
    all_aliases = set(already_alternate_names)
    for dp in nodes_data:
        dp_alts = dp.get("alternate_names", [])
        if isinstance(dp_alts, list):
            all_aliases.update(dp_alts)
        elif isinstance(dp_alts, str) and dp_alts:
            all_aliases.update([a.strip() for a in dp_alts.split(",") if a.strip()])
    all_aliases.discard(entity_name)
    clean_alternate_names = sorted(all_aliases)

    # Merge time ranges
    all_time_ranges = sorted(set(
        filter(None, [dp.get("time_range") for dp in nodes_data] + already_time_ranges)
    ))
    time_range = GRAPH_FIELD_SEP.join(all_time_ranges) if all_time_ranges else "unknown"

    # Merge geo locations
    all_geo_locations = sorted(set(
        filter(None, [dp.get("geo_location") for dp in nodes_data] + already_geo_locations)
    ))
    geo_location = GRAPH_FIELD_SEP.join(all_geo_locations) if all_geo_locations else "unknown"

    kwargs = {
        "entity_name": entity_name,
        "entity_type": entity_type,
        "description": description,
        "source_id": source_id,
        "alternate_names": clean_alternate_names,
        "time_range": time_range if time_range != "unknown" else None,
        "geo_location": geo_location if geo_location != "unknown" else None,
    }
    ent_type_lower = entity_type.lower()
    if ent_type_lower == "person":
        inst = Person(**kwargs)
    elif ent_type_lower == "place":
        inst = Place(**kwargs)
    elif ent_type_lower == "time":
        inst = TimeEntity(**kwargs)
    elif ent_type_lower == "event":
        inst = Event(**kwargs)
    elif ent_type_lower == "artifact":
        inst = Artifact(**kwargs)
    elif ent_type_lower == "object":
        inst = ObjectEntity(**kwargs)
    elif ent_type_lower == "concept":
        inst = Concept(**kwargs)
    else:
        inst = BaseEntity(**kwargs)
        
    node_data = _sanitize_for_graphml(inst.model_dump())
    node_data["entity_type"] = entity_type
    node_data["alternate_names"] = clean_alternate_names
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_keywords = []
    already_time_ranges = []

    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_keywords.extend(
            split_string_by_multi_markers(already_edge["keywords"], [GRAPH_FIELD_SEP])
        )
        if "time_range" in already_edge and already_edge["time_range"]:
            already_time_ranges.append(already_edge["time_range"])

    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    keywords = GRAPH_FIELD_SEP.join(
        sorted(set([dp["keywords"] for dp in edges_data] + already_keywords))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )

    # Merge relationship time ranges
    all_time_ranges = sorted(set(
        filter(None, [dp.get("time_range") for dp in edges_data] + already_time_ranges)
    ))
    time_range = GRAPH_FIELD_SEP.join(all_time_ranges) if all_time_ranges else "unknown"

    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            inst = BaseEntity(
                entity_name=need_insert_id,
                source_id=source_id,
                description=description,
                entity_type="UNKNOWN",
                time_range=None,
                geo_location=None,
            )
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data=_sanitize_for_graphml(inst.model_dump()),
            )

    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=_sanitize_for_graphml(dict(
            weight=weight,
            description=description,
            keywords=keywords,
            source_id=source_id,
            time_range=time_range,
        )),
    )

    edge_data = _sanitize_for_graphml(dict(
        src_id=src_id,
        tgt_id=tgt_id,
        description=description,
        keywords=keywords,
        time_range=time_range,
    ))

    return edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    retry_max = int(os.getenv("LLM_EXTRACT_MAX_RETRIES", "2"))
    backoff_ms = float(os.getenv("LLM_EXTRACT_RETRY_BACKOFF_MS", "500"))
    skip_on_error = os.getenv("LLM_EXTRACT_SKIP_ON_ERROR", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    ordered_chunks = list(chunks.items())
    use_minirag_prompt = global_config.get("use_minirag_prompt", False)
    if use_minirag_prompt:
        from .prompt_minirag import PROMPTS as MINIRAG_PROMPTS
        target_prompts = MINIRAG_PROMPTS
    else:
        target_prompts = PROMPTS

    entity_extract_prompt = target_prompts["entity_extraction"]

    context_base = dict(
        tuple_delimiter=target_prompts["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=target_prompts["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=target_prompts["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(target_prompts["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = target_prompts["entiti_continue_extraction"]

    if_loop_prompt = target_prompts["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _call_llm(prompt: str, chunk_key: str, history_messages=None):
        attempts = 0
        while True:
            try:
                with _suppress_httpx_logs():
                    if history_messages is not None:
                        return await use_llm_func(
                            prompt, history_messages=history_messages
                        )
                    return await use_llm_func(prompt)
            except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
                attempts += 1
                if attempts > retry_max:
                    if skip_on_error:
                        logger.warning(
                            "LLM call failed after %s attempts; skipping chunk %s: %s",
                            attempts,
                            chunk_key,
                            exc,
                        )
                        return None
                    raise
                backoff_seconds = (backoff_ms / 1000.0) * attempts
                logger.warning(
                    "LLM retry %s/%s for chunk %s after error: %s (backoff %.2fs)",
                    attempts,
                    retry_max,
                    chunk_key,
                    exc,
                    backoff_seconds,
                )
                await asyncio.sleep(backoff_seconds)

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await _call_llm(hint_prompt, chunk_key)
        if final_result is None:
            return {}, {}

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await _call_llm(
                continue_prompt, chunk_key, history_messages=history
            )
            if glean_result is None:
                break

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result = await _call_llm(
                if_loop_prompt, chunk_key, history_messages=history
            )
            if if_loop_result is None:
                break
            if_loop_result = str(if_loop_result)
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key, content
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        print(
            f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r",
            end="",
            flush=False,
        )
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    print(f"Starting entity and relationship extraction on {len(ordered_chunks)} chunks...")
    
    # 1. Parallel raw entity/relation extraction via LLM
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks], return_exceptions=True
    )
    print()  # clear progress/status info
    print(f"Finished raw LLM entity and relationship extraction. Total processed chunks: {already_processed}")

    # 2. KG merging with fuzzy logic
    enable_fl = global_config.get("enable_fuzzy_logic", False)
    if enable_fl:
        from .fuzzy_alignment import FloraEntityAligner
        aligner = FloraEntityAligner(
            global_config=global_config,
            embedding_func=global_config.get("embedding_func"),
        )
        print("Using FLORA fuzzy logic node alignment to reduce duplicate nodes...")
        # Sequential queue logic for small KG alignment
        for result in results:
            if isinstance(result, Exception) or not result:
                continue
            m_nodes, m_edges = result
            if not m_nodes and not m_edges:
                continue

            aligned_nodes, aligned_edges = await aligner.align(
                m_nodes, m_edges, knowledge_graph_inst
            )

            # Sequentially merge this aligned small KG into the core graph storage
            all_entities_data = await asyncio.gather(
                *[
                    _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                    for k, v in aligned_nodes.items()
                ]
            )
            all_relationships_data = await asyncio.gather(
                *[
                    _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
                    for k, v in aligned_edges.items()
                ]
            )

            # Index merged entities/relations to VDBs immediately so next sequential steps can search them
            if all_entities_data and entity_vdb is not None:
                data_for_vdb = {}
                for dp in all_entities_data:
                    alts_str = ", ".join(dp["alternate_names"]) if dp.get("alternate_names") else ""
                    content_str = f"{dp['entity_name']} {f'(Tên khác: {alts_str})' if alts_str else ''} {dp['description']}"
                    data_for_vdb[compute_mdhash_id(dp["entity_name"], prefix="ent-")] = {
                        "content": content_str,
                        "entity_name": dp["entity_name"],
                    }
                await entity_vdb.upsert(data_for_vdb)
            if all_entities_data and entity_name_vdb is not None:
                data_for_vdb = {}
                for dp in all_entities_data:
                    data_for_vdb[compute_mdhash_id(dp["entity_name"], prefix="Ename-")] = {
                        "content": dp["entity_name"],
                        "entity_name": dp["entity_name"],
                    }
                    for alt in dp.get("alternate_names", []):
                        if alt and alt != dp["entity_name"]:
                            data_for_vdb[compute_mdhash_id(f"{dp['entity_name']}_{alt}", prefix="EnameAlt-")] = {
                                "content": alt,
                                "entity_name": dp["entity_name"],
                            }
                await entity_name_vdb.upsert(data_for_vdb)
            if all_relationships_data and relationships_vdb is not None:
                data_for_vdb = {
                    compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                        "src_id": dp["src_id"],
                        "tgt_id": dp["tgt_id"],
                        "content": dp["keywords"]
                        + " " + dp["src_id"]
                        + " " + dp["tgt_id"]
                        + " " + dp["description"],
                    }
                    for dp in all_relationships_data
                }
                await relationships_vdb.upsert(data_for_vdb)

    else:
        print()  # clear the progress bar
        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for m_nodes, m_edges in results:
            for k, v in m_nodes.items():
                maybe_nodes[k].extend(v)
            for k, v in m_edges.items():
                maybe_edges[tuple(sorted(k))].extend(v)
        all_entities_data = await asyncio.gather(
            *[
                _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
                for k, v in maybe_nodes.items()
            ]
        )
        all_relationships_data = await asyncio.gather(
            *[
                _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
                for k, v in maybe_edges.items()
            ]
        )
        if not len(all_entities_data):
            logger.warning("Didn't extract any entities, maybe your LLM is not working")
            return None
        if not len(all_relationships_data):
            logger.warning(
                "Didn't extract any relationships, maybe your LLM is not working"
            )
            return None

        if entity_vdb is not None:
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + dp["description"],
                    "entity_name": dp["entity_name"],
                }
                for dp in all_entities_data
            }
            await entity_vdb.upsert(data_for_vdb)
        if entity_vdb is not None:
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                    "content": dp["entity_name"] + " " + dp["description"],
                    "entity_name": dp["entity_name"],
                }
                for dp in all_entities_data
            }
            await entity_vdb.upsert(data_for_vdb)

        if entity_name_vdb is not None:
            data_for_vdb = {
                compute_mdhash_id(dp["entity_name"], prefix="Ename-"): {
                    "content": dp["entity_name"],
                    "entity_name": dp["entity_name"],
                }
                for dp in all_entities_data
            }
            await entity_name_vdb.upsert(data_for_vdb)

        if relationships_vdb is not None:
            data_for_vdb = {
                compute_mdhash_id(dp["src_id"] + dp["tgt_id"], prefix="rel-"): {
                    "src_id": dp["src_id"],
                    "tgt_id": dp["tgt_id"],
                    "content": dp["keywords"]
                    + " " + dp["src_id"]
                    + " " + dp["tgt_id"]
                    + " " + dp["description"],
                }
                for dp in all_relationships_data
            }

            await relationships_vdb.upsert(data_for_vdb)

    return knowledge_graph_inst


async def local_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt, keyword_extraction=True)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        keywords = keywords_data.get("low_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("low_level_keywords", [])
            keywords = ", ".join(keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        context = await _build_local_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )
    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]
    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt) and response.startswith(sys_prompt):
        response = response[len(sys_prompt):].strip()

    return response


async def _build_local_query_context(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await entities_vdb.query(query, top_k=query_param.top_k)

    if not len(results):
        return None
    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(r["entity_name"]) for r in results]
    )
    if not all([n is not None for n in node_datas]):
        logger.warning("Some nodes are missing, maybe the storage is damaged")
    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in results]
    )
    node_datas = [
        {**n, "entity_name": k["entity_name"], "rank": d}
        for k, n, d in zip(results, node_datas, node_degrees)
        if n is not None
    ]  # what is this text_chunks_db doing.  dont remember it in airvx.  check the diagram.
    use_text_units = await _find_most_related_text_unit_from_entities(
        node_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    use_relations = await _find_most_related_edges_from_entities(
        node_datas, query_param, knowledge_graph_inst
    )
    logger.info(
        f"Local query uses {len(node_datas)} entites, {len(use_relations)} relations, {len(use_text_units)} text units"
    )
    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(use_relations):
        relations_section_list.append(
            [
                i,
                e["src_tgt"][0],
                e["src_tgt"][1],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)
    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _find_most_related_text_unit_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])

    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )

    # Add null check for node data
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None and "source_id" in v  # Add source_id check
    }

    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            if this_edges:  # Add check for None edges
                for e in this_edges:
                    if (
                        e[1] in all_one_hop_text_units_lookup
                        and c_id in all_one_hop_text_units_lookup[e[1]]
                    ):
                        relation_counts += 1

            chunk_data = await text_chunks_db.get_by_id(c_id)
            if chunk_data is not None and "content" in chunk_data:  # Add content check
                all_text_units_lookup[c_id] = {
                    "data": chunk_data,
                    "order": index,
                    "relation_counts": relation_counts,
                }

    # Filter out None values and ensure data has content
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and "content" in v["data"]
    ]

    if not all_text_units:
        logger.warning("No valid text units found")
        return []

    all_text_units = sorted(
        all_text_units, key=lambda x: (x["order"], -x["relation_counts"])
    )

    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )

    all_text_units = [t["data"] for t in all_text_units]
    return all_text_units


async def _find_most_related_edges_from_entities(
    node_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    all_related_edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_edges = set()
    for this_edges in all_related_edges:
        all_edges.update([tuple(sorted(e)) for e in this_edges])
    all_edges = list(all_edges)
    all_edges_pack = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(e[0], e[1]) for e in all_edges]
    )
    all_edges_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(e[0], e[1]) for e in all_edges]
    )
    all_edges_data = [
        {"src_tgt": k, "rank": d, **v}
        for k, v, d in zip(all_edges, all_edges_pack, all_edges_degree)
        if v is not None
    ]
    all_edges_data = sorted(
        all_edges_data, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    all_edges_data = truncate_list_by_token_size(
        all_edges_data,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )
    return all_edges_data


async def global_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)
    result = await use_model_func(kw_prompt, keyword_extraction=True)
    json_text = locate_json_string_body_from_string(result)

    try:
        keywords_data = json.loads(json_text)
        keywords = keywords_data.get("high_level_keywords", [])
        keywords = ", ".join(keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"

            keywords_data = json.loads(result)
            keywords = keywords_data.get("high_level_keywords", [])
            keywords = ", ".join(keywords)

        except json.JSONDecodeError as e:
            # Handle parsing error
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if keywords:
        context = await _build_global_query_context(
            keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt) and response.startswith(sys_prompt):
        response = response[len(sys_prompt):].strip()

    return response


async def _build_global_query_context(
    keywords,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
):
    results = await relationships_vdb.query(keywords, top_k=query_param.top_k)

    if not len(results):
        return None

    edge_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_edge(r["src_id"], r["tgt_id"]) for r in results]
    )

    if not all([n is not None for n in edge_datas]):
        logger.warning("Some edges are missing, maybe the storage is damaged")
    edge_degree = await asyncio.gather(
        *[knowledge_graph_inst.edge_degree(r["src_id"], r["tgt_id"]) for r in results]
    )
    edge_datas = [
        {"src_id": k["src_id"], "tgt_id": k["tgt_id"], "rank": d, **v}
        for k, v, d in zip(results, edge_datas, edge_degree)
        if v is not None
    ]
    edge_datas = sorted(
        edge_datas, key=lambda x: (x["rank"], x["weight"]), reverse=True
    )
    edge_datas = truncate_list_by_token_size(
        edge_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_global_context,
    )

    use_entities = await _find_most_related_entities_from_relationships(
        edge_datas, query_param, knowledge_graph_inst
    )
    use_text_units = await _find_related_text_unit_from_relationships(
        edge_datas, query_param, text_chunks_db, knowledge_graph_inst
    )
    logger.info(
        f"Global query uses {len(use_entities)} entites, {len(edge_datas)} relations, {len(use_text_units)} text units"
    )
    relations_section_list = [
        ["id", "source", "target", "description", "keywords", "weight", "rank"]
    ]
    for i, e in enumerate(edge_datas):
        relations_section_list.append(
            [
                i,
                e["src_id"],
                e["tgt_id"],
                e["description"],
                e["keywords"],
                e["weight"],
                e["rank"],
            ]
        )
    relations_context = list_of_list_to_csv(relations_section_list)

    entites_section_list = [["id", "entity", "type", "description", "rank"]]
    for i, n in enumerate(use_entities):
        entites_section_list.append(
            [
                i,
                n["entity_name"],
                n.get("entity_type", "UNKNOWN"),
                n.get("description", "UNKNOWN"),
                n["rank"],
            ]
        )
    entities_context = list_of_list_to_csv(entites_section_list)

    text_units_section_list = [["id", "content"]]
    for i, t in enumerate(use_text_units):
        text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Relationships-----
```csv
{relations_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def _find_most_related_entities_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    knowledge_graph_inst: BaseGraphStorage,
):
    entity_names = set()
    for e in edge_datas:
        entity_names.add(e["src_id"])
        entity_names.add(e["tgt_id"])

    node_datas = await asyncio.gather(
        *[knowledge_graph_inst.get_node(entity_name) for entity_name in entity_names]
    )

    node_degrees = await asyncio.gather(
        *[knowledge_graph_inst.node_degree(entity_name) for entity_name in entity_names]
    )
    node_datas = [
        {**n, "entity_name": k, "rank": d}
        for k, n, d in zip(entity_names, node_datas, node_degrees)
    ]

    node_datas = truncate_list_by_token_size(
        node_datas,
        key=lambda x: x["description"],
        max_token_size=query_param.max_token_for_local_context,
    )

    return node_datas


async def _find_related_text_unit_from_relationships(
    edge_datas: list[dict],
    query_param: QueryParam,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in edge_datas
    ]

    all_text_units_lookup = {}

    for index, unit_list in enumerate(text_units):
        for c_id in unit_list:
            if c_id not in all_text_units_lookup:
                all_text_units_lookup[c_id] = {
                    "data": await text_chunks_db.get_by_id(c_id),
                    "order": index,
                }

    if any([v is None or v.get("data") is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v}
        for k, v in all_text_units_lookup.items()
        if v is not None and v.get("data") is not None and v["data"].get("content") is not None
    ]
    all_text_units = sorted(all_text_units, key=lambda x: x["order"])
    all_text_units = truncate_list_by_token_size(
        all_text_units,
        key=lambda x: x["data"]["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    all_text_units: list[TextChunkSchema] = [t["data"] for t in all_text_units]

    return all_text_units


async def hybrid_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    low_level_context = None
    high_level_context = None
    use_model_func = global_config["llm_model_func"]

    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)

    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)
    try:
        keywords_data = json.loads(json_text)
        hl_keywords = keywords_data.get("high_level_keywords", [])
        ll_keywords = keywords_data.get("low_level_keywords", [])
        hl_keywords = ", ".join(hl_keywords)
        ll_keywords = ", ".join(ll_keywords)
    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            keywords_data = json.loads(result)
            hl_keywords = keywords_data.get("high_level_keywords", [])
            ll_keywords = keywords_data.get("low_level_keywords", [])
            hl_keywords = ", ".join(hl_keywords)
            ll_keywords = ", ".join(ll_keywords)
        # Handle parsing error
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]
    if ll_keywords:
        low_level_context = await _build_local_query_context(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )

    if hl_keywords:
        high_level_context = await _build_global_query_context(
            hl_keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )

    context = combine_contexts(high_level_context, low_level_context)

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt) and response.startswith(sys_prompt):
        response = response[len(sys_prompt):].strip()
    return response


def combine_contexts(high_level_context, low_level_context):
    # Function to extract entities, relationships, and sources from context strings

    def extract_sections(context):
        entities_match = re.search(
            r"-----Entities-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        relationships_match = re.search(
            r"-----Relationships-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )
        sources_match = re.search(
            r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, re.DOTALL
        )

        entities = entities_match.group(1) if entities_match else ""
        relationships = relationships_match.group(1) if relationships_match else ""
        sources = sources_match.group(1) if sources_match else ""

        return entities, relationships, sources

    # Extract sections from both contexts

    if high_level_context is None:
        warnings.warn(
            "High Level context is None. Return empty High entity/relationship/source"
        )
        hl_entities, hl_relationships, hl_sources = "", "", ""
    else:
        hl_entities, hl_relationships, hl_sources = extract_sections(high_level_context)

    if low_level_context is None:
        warnings.warn(
            "Low Level context is None. Return empty Low entity/relationship/source"
        )
        ll_entities, ll_relationships, ll_sources = "", "", ""
    else:
        ll_entities, ll_relationships, ll_sources = extract_sections(low_level_context)

    # Combine and deduplicate the entities

    combined_entities = process_combine_contexts(hl_entities, ll_entities)
    combined_entities = chunking_by_token_size(combined_entities, max_token_size=2000)
    # Combine and deduplicate the relationships
    combined_relationships = process_combine_contexts(
        hl_relationships, ll_relationships
    )
    combined_relationships = chunking_by_token_size(
        combined_relationships, max_token_size=2000
    )
    # Combine and deduplicate the sources
    combined_sources = process_combine_contexts(hl_sources, ll_sources)
    combined_sources = chunking_by_token_size(combined_sources, max_token_size=2000)
    # Format the combined context
    return f"""
-----Entities-----
```csv
{combined_entities}
```
-----Relationships-----
```csv
{combined_relationships}
```
-----Sources-----
```csv
{combined_sources}
```
"""


async def naive_query(
    query,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["llm_model_func"]
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    if not len(results):
        return PROMPTS["fail_response"]
    chunks_ids = [r["id"] for r in results]

    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "--New Chunk--\n".join([c["content"] for c in maybe_trun_chunks])
    if query_param.only_need_context:
        return section
    sys_prompt_temp = PROMPTS["naive_rag_response"]
    sys_prompt = sys_prompt_temp.format(
        content_data=section, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    if len(response) > len(sys_prompt) and response.startswith(sys_prompt):
        response = response[len(sys_prompt):].strip()
    return response

    return response


async def path2chunk(
    scored_edged_reasoning_path, knowledge_graph_inst, pairs_append, query, max_chunks=5
):
    already_node = {}
    for k, v in scored_edged_reasoning_path.items():
        node_chunk_id = None

        for pathtuple, scorelist in v["Path"].items():
            if pathtuple in pairs_append:
                use_edge = pairs_append[pathtuple]
                edge_datas = []
                edge_datas = await asyncio.gather(
                    *[knowledge_graph_inst.get_edge(r[0], r[1]) for r in use_edge]
                )
                text_units = [
                    split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
                    for dp in edge_datas  # chunk ID
                ][0]

            else:
                use_edge = []
                text_units = []

            node_datas = await asyncio.gather(
                *[knowledge_graph_inst.get_node(pathtuple[0])]
            )
            for dp in node_datas:
                text_units_node = split_string_by_multi_markers(
                    dp["source_id"], [GRAPH_FIELD_SEP]
                )
                text_units = text_units + text_units_node

            node_datas = await asyncio.gather(
                *[knowledge_graph_inst.get_node(ents) for ents in pathtuple[1:]]
            )
            if query is not None:
                for dp in node_datas:
                    text_units_node = split_string_by_multi_markers(
                        dp["source_id"], [GRAPH_FIELD_SEP]
                    )
                    descriptionlist_node = split_string_by_multi_markers(
                        dp["description"], [GRAPH_FIELD_SEP]
                    )
                    if descriptionlist_node[0] not in already_node.keys():
                        already_node[descriptionlist_node[0]] = None

                        if len(text_units_node) == len(descriptionlist_node):
                            if len(text_units_node) > 5:
                                max_ids = int(max(5, len(text_units_node) / 2))
                                should_consider_idx = calculate_similarity(
                                    descriptionlist_node, query, k=max_ids
                                )
                                text_units_node = [
                                    text_units_node[i] for i in should_consider_idx
                                ]
                                already_node[descriptionlist_node[0]] = text_units_node
                    else:
                        text_units_node = already_node[descriptionlist_node[0]]
                    if text_units_node is not None:
                        text_units = text_units + text_units_node

            count_dict = Counter(text_units)
            total_score = scorelist[0] + scorelist[1] + 1
            for key, value in count_dict.items():
                count_dict[key] = value * total_score
            if node_chunk_id is None:
                node_chunk_id = count_dict
            else:
                node_chunk_id = node_chunk_id + count_dict
        v["Path"] = []
        if node_chunk_id is None:
            node_datas = await asyncio.gather(*[knowledge_graph_inst.get_node(k)])
            for dp in node_datas:
                text_units_node = split_string_by_multi_markers(
                    dp["source_id"], [GRAPH_FIELD_SEP]
                )
                count_dict = Counter(text_units_node)

            for id in count_dict.most_common(max_chunks):
                v["Path"].append(id[0])
            # v['Path'] = count_dict.most_common(max_chunks)#[]
        else:
            for id in count_dict.most_common(max_chunks):
                v["Path"].append(id[0])
            # v['Path'] = node_chunk_id.most_common(max_chunks)
    return scored_edged_reasoning_path


def scorednode2chunk(input_dict, values_dict):
    for key, value_list in input_dict.items():
        input_dict[key] = [
            values_dict.get(val, None) for val in value_list if val in values_dict
        ]
        input_dict[key] = [val for val in input_dict[key] if val is not None]


def kwd2chunk(ent_from_query_dict, chunks_ids, chunk_nums):
    final_chunk = Counter()
    final_chunk_id = []
    for key, list_of_dicts in ent_from_query_dict.items():
        total_id_scores = Counter()
        id_scores_list = []
        id_scores = {}
        for d in list_of_dicts:
            if d == list_of_dicts[0]:
                score = d["Score"] * 2
            else:
                score = d["Score"]
            path = d["Path"]

            for id in path:
                if id == path[0] and id in chunks_ids:
                    score = score * 10
                if id in id_scores:
                    id_scores[id] += score
                else:
                    id_scores[id] = score
        id_scores_list.append(id_scores)

        for scores in id_scores_list:
            total_id_scores.update(scores)
        final_chunk = final_chunk + total_id_scores  # .most_common(3)

    for i in final_chunk.most_common(chunk_nums):
        final_chunk_id.append(i[0])
    return final_chunk_id


async def _build_mini_query_context(
    ent_from_query,
    type_keywords,
    originalquery,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    embedder,
    query_param: QueryParam,
):
    imp_ents = []
    nodes_from_query_list = []
    ent_from_query_dict = {}

    for ent in ent_from_query:
        ent_from_query_dict[ent] = []
        results_node = await entity_name_vdb.query(ent, top_k=query_param.top_k)

        nodes_from_query_list.append(results_node)
        ent_from_query_dict[ent] = [e["entity_name"] for e in results_node]

    candidate_reasoning_path = {}

    for results_node_list in nodes_from_query_list:
        candidate_reasoning_path_new = {
            key["entity_name"]: {"Score": key["distance"], "Path": []}
            for key in results_node_list
        }

        candidate_reasoning_path = {
            **candidate_reasoning_path,
            **candidate_reasoning_path_new,
        }
    for key in candidate_reasoning_path.keys():
        candidate_reasoning_path[key][
            "Path"
        ] = await knowledge_graph_inst.get_neighbors_within_k_hops(key, 2)
        imp_ents.append(key)

    short_path_entries = {
        name: entry
        for name, entry in candidate_reasoning_path.items()
        if len(entry["Path"]) < 1
    }
    sorted_short_path_entries = sorted(
        short_path_entries.items(), key=lambda x: x[1]["Score"], reverse=True
    )
    save_p = max(1, int(len(sorted_short_path_entries) * 0.2))
    top_short_path_entries = sorted_short_path_entries[:save_p]
    top_short_path_dict = {name: entry for name, entry in top_short_path_entries}
    long_path_entries = {
        name: entry
        for name, entry in candidate_reasoning_path.items()
        if len(entry["Path"]) >= 1
    }
    candidate_reasoning_path = {**long_path_entries, **top_short_path_dict}
    node_datas_from_type = await knowledge_graph_inst.get_node_from_types(
        type_keywords
    )  # entity_type, description,...

    maybe_answer_list = [n["entity_name"] for n in node_datas_from_type]
    imp_ents = imp_ents + maybe_answer_list
    scored_reasoning_path = cal_path_score_list(
        candidate_reasoning_path, maybe_answer_list
    )

    results_edge = await relationships_vdb.query(
        originalquery, top_k=len(ent_from_query) * query_param.top_k
    )
    goodedge = []
    badedge = []
    for item in results_edge:
        if item["src_id"] in imp_ents or item["tgt_id"] in imp_ents:
            goodedge.append(item)
        else:
            badedge.append(item)
    scored_edged_reasoning_path, pairs_append = edge_vote_path(
        scored_reasoning_path, goodedge
    )
    scored_edged_reasoning_path = await path2chunk(
        scored_edged_reasoning_path,
        knowledge_graph_inst,
        pairs_append,
        originalquery,
        max_chunks=3,
    )

    entites_section_list = []
    node_datas = await asyncio.gather(
        *[
            knowledge_graph_inst.get_node(entity_name)
            for entity_name in scored_edged_reasoning_path.keys()
        ]
    )
    node_datas = [
        {**n, "entity_name": k, "Score": scored_edged_reasoning_path[k]["Score"]}
        for k, n in zip(scored_edged_reasoning_path.keys(), node_datas)
    ]
    for i, n in enumerate(node_datas):
        entites_section_list.append(
            [
                n["entity_name"],
                n["Score"],
                n.get("description", "UNKNOWN"),
            ]
        )
    entites_section_list = sorted(
        entites_section_list, key=lambda x: x[1], reverse=True
    )
    entites_section_list = truncate_list_by_token_size(
        entites_section_list,
        key=lambda x: x[2],
        max_token_size=query_param.max_token_for_node_context,
    )

    entites_section_list.insert(0, ["entity", "score", "description"])
    entities_context = list_of_list_to_csv(entites_section_list)

    scorednode2chunk(ent_from_query_dict, scored_edged_reasoning_path)

    results = await chunks_vdb.query(originalquery, top_k=int(query_param.top_k / 2))
    chunks_ids = [r["id"] for r in results]
    final_chunk_id = kwd2chunk(
        ent_from_query_dict, chunks_ids, chunk_nums=int(query_param.top_k / 2)
    )

    if not len(results_node):
        return None

    if not len(results_edge):
        return None

    use_text_units = await asyncio.gather(
        *[text_chunks_db.get_by_id(id) for id in final_chunk_id]
    )
    text_units_section_list = [["id", "content"]]

    for i, t in enumerate(use_text_units):
        if t is not None:
            text_units_section_list.append([i, t["content"]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    return f"""
-----Entities-----
```csv
{entities_context}
```
-----Sources-----
```csv
{text_units_context}
```
"""


async def minirag_query(  # MiniRAG
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    entity_name_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    chunks_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    embedder,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    use_model_func = global_config["llm_model_func"]
    kw_prompt_temp = PROMPTS["minirag_query2kwd"]
    TYPE_POOL, TYPE_POOL_w_CASE = await knowledge_graph_inst.get_types()
    kw_prompt = kw_prompt_temp.format(query=query, TYPE_POOL=TYPE_POOL)
    result = await use_model_func(kw_prompt)

    try:
        keywords_data = json_repair.loads(result)

        type_keywords = keywords_data.get("answer_type_keywords", [])
        entities_from_query = keywords_data.get("entities_from_query", [])[:5]

    except json.JSONDecodeError:
        try:
            result = (
                result.replace(kw_prompt[:-1], "")
                .replace("user", "")
                .replace("model", "")
                .strip()
            )
            result = "{" + result.split("{")[1].split("}")[0] + "}"
            keywords_data = json_repair.loads(result)
            type_keywords = keywords_data.get("answer_type_keywords", [])
            entities_from_query = keywords_data.get("entities_from_query", [])[:5]

        # Handle parsing error
        except Exception as e:
            print(f"JSON parsing error: {e}")
            return PROMPTS["fail_response"]

    context = await _build_mini_query_context(
        entities_from_query,
        type_keywords,
        query,
        knowledge_graph_inst,
        entities_vdb,
        entity_name_vdb,
        relationships_vdb,
        chunks_vdb,
        text_chunks_db,
        embedder,
        query_param,
    )

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    sys_prompt_temp = PROMPTS["rag_response"]
    sys_prompt = sys_prompt_temp.format(
        context_data=context, response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )

    return response



async def simgrag_query(
    query,
    knowledge_graph_inst: BaseGraphStorage,
    entities_vdb: BaseVectorStorage,
    relationships_vdb: BaseVectorStorage,
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """
    Simple KG-RAG mode (simrag):
    1. Extract high-level and low-level keywords from query using LLM
    2. Retrieve candidate entities, relationships, and text chunks from KG
    3. Combine contexts into a compact summary prompt and ask LLM to synthesize a concise answer
    """
    low_level_context = None
    high_level_context = None
    use_model_func = global_config["llm_model_func"]

    # Step 1: Extract keywords
    kw_prompt_temp = PROMPTS["keywords_extraction"]
    kw_prompt = kw_prompt_temp.format(query=query)

    result = await use_model_func(kw_prompt)
    json_text = locate_json_string_body_from_string(result)
    hl_keywords, ll_keywords = "", ""
    try:
        if json_text is None:
            raise ValueError("json_text is None")
        keywords_data = json.loads(json_text)
        hl_keywords = ", ".join(keywords_data.get("high_level_keywords", []))
        ll_keywords = ", ".join(keywords_data.get("low_level_keywords", []))
    except Exception:
        try:
            if result and "{" in result and "}" in result:
                clean_result = "{" + result.split("{")[1].split("}")[0] + "}"
                keywords_data = json.loads(clean_result)
                hl_keywords = ", ".join(keywords_data.get("high_level_keywords", []))
                ll_keywords = ", ".join(keywords_data.get("low_level_keywords", []))
            else:
                hl_keywords = query
                ll_keywords = query
        except Exception as e:
            logger.warning(f"Keyword extraction fallback failed: {e}")
            hl_keywords = query
            ll_keywords = query

    # Fallback if both keywords are empty
    if not ll_keywords and not hl_keywords:
        ll_keywords = query

    # Step 2: Retrieve candidates from KG
    if ll_keywords:
        low_level_context = await _build_local_query_context(
            ll_keywords,
            knowledge_graph_inst,
            entities_vdb,
            text_chunks_db,
            query_param,
        )

    if hl_keywords:
        high_level_context = await _build_global_query_context(
            hl_keywords,
            knowledge_graph_inst,
            entities_vdb,
            relationships_vdb,
            text_chunks_db,
            query_param,
        )

    context = combine_contexts(high_level_context, low_level_context)

    if query_param.only_need_context:
        return context
    if context is None:
        return PROMPTS["fail_response"]

    # Step 3: Ask LLM for concise synthesis
    sys_prompt = f"""---Role---
You are a concise AI assistant. Answer the user's question directly and concisely based ONLY on the provided Knowledge Graph context data.

---Knowledge Graph Context Data---
{context}

---Instructions---
- Provide a direct, factual, and compact answer.
- Do not output meta-talk, unnecessary intro, or markdown filler.
"""
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
    )
    if len(response) > len(sys_prompt) and response.startswith(sys_prompt):
        response = response[len(sys_prompt):].strip()
    return response

