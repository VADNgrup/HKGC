import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Set

from .schema import parse_time_range


def parse_entity_tokens(name: str, alternate_names: Optional[List[str]]) -> Set[str]:
    """Extracts uppercase normalized word tokens from a name and its alternate names."""
    all_names = [name]
    if alternate_names:
        if isinstance(alternate_names, list):
            all_names.extend(alternate_names)
        elif isinstance(alternate_names, str):
            all_names.extend([a.strip() for a in alternate_names.split(",") if a.strip()])

    tokens = set()
    for n in all_names:
        if n:
            tokens.update(re.findall(r"\w+", str(n).upper()))
    return tokens


def _identity_similarity(
    name_a: str, alts_a: Optional[List[str]], name_b: str, alts_b: Optional[List[str]]
) -> float:
    """
    Computes Identity Similarity between two nodes A and B:
    max( |A_tokens ∩ B_tokens| / |A_tokens|, |A_tokens ∩ B_tokens| / |B_tokens| )
    """
    tokens_a = parse_entity_tokens(name_a, alts_a)
    tokens_b = parse_entity_tokens(name_b, alts_b)

    if not tokens_a or not tokens_b:
        return 0.0

    inter = len(tokens_a & tokens_b)
    sim_a = inter / len(tokens_a)
    sim_b = inter / len(tokens_b)
    return max(sim_a, sim_b)


def _neighbourhood_similarity(neighbors_a: Set[str], neighbors_b: Set[str]) -> float:
    """
    Computes Neighbourhood Similarity between two nodes A and B:
    max( |A_neigh ∩ B_neigh| / |A_neigh|, |A_neigh ∩ B_neigh| / |B_neigh| )
    """
    if not neighbors_a or not neighbors_b:
        return 0.0

    inter = len(neighbors_a & neighbors_b)
    sim_a = inter / len(neighbors_a)
    sim_b = inter / len(neighbors_b)
    return max(sim_a, sim_b)


def _temporal_status(time_a: Optional[str], time_b: Optional[str]) -> str:
    """
    Determines time relationship between time_a and time_b.
    Returns:
        "no_info" if either time is unknown/unparseable.
        "overlap" if both time ranges overlap.
        "mismatch" if both are known but do not overlap.
    """
    range_a = parse_time_range(time_a)
    range_b = parse_time_range(time_b)

    if range_a is None or range_b is None:
        return "no_info"

    start_a, end_a = range_a
    start_b, end_b = range_b

    if max(start_a, start_b) <= min(end_a, end_b):
        return "overlap"
    return "mismatch"


class FloraEntityAligner:
    """
    Simplified Knowledge Graph Entity Alignment.

    Decision Logic for nodes A & B:
      - identity_similarity: token overlap of (name + alternate_names)
      - neighbourhood_similarity: ratio overlap of graph neighbors
      - time_status: overlap / no_info / mismatch

      ACCEPT if:
        1. identity_similarity >= thres and time_status == 'overlap'
        2. identity_similarity >= thres and time_status == 'no_info'
        3. neighbourhood_similarity >= thres
    """

    def __init__(
        self,
        global_config: dict,
        embedding_func=None,
        thres_sim: float | None = None,
        thres_map: float | None = None,
        use_llm_fuzzy: bool | None = None,
    ):
        self.global_config = global_config
        self.embedding_func = embedding_func

        # Load threshold settings
        self.thres_sim = (
            thres_sim
            if thres_sim is not None
            else float(os.getenv("FL_THRES_SIM", "0.75"))
        )
        self.thres_map = (
            thres_map
            if thres_map is not None
            else float(os.getenv("FL_THRES_MAP", "0.75"))
        )

        if use_llm_fuzzy is not None:
            self.use_llm_fuzzy = use_llm_fuzzy
        else:
            self.use_llm_fuzzy = global_config.get(
                "use_llm_fuzzy",
                os.getenv("FL_USE_LLM", "0").lower() in {"1", "true", "yes", "on"},
            )

    def _cluster_local_kg(self, m_nodes: dict, m_edges: dict) -> Tuple[dict, dict]:
        """
        Deduplicates entities within a single chunk using Union-Find based on high identity similarity.
        """
        entities_set = set(m_nodes.keys())
        for src, tgt in m_edges.keys():
            entities_set.add(src)
            entities_set.add(tgt)

        entities = list(entities_set)
        if not entities:
            return m_nodes, m_edges

        parent = {e: e for e in entities}

        def find(n):
            if parent[n] == n:
                return n
            parent[n] = find(parent[n])
            return parent[n]

        def union(n1, n2):
            r1 = find(n1)
            r2 = find(n2)
            if r1 != r2:
                if len(r1) >= len(r2):
                    parent[r2] = r1
                else:
                    parent[r1] = r2

        # Cluster similar nodes within chunk
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                ent1, ent2 = entities[i], entities[j]
                alts1 = m_nodes.get(ent1, [{}])[0].get("alternate_names", []) if ent1 in m_nodes else []
                alts2 = m_nodes.get(ent2, [{}])[0].get("alternate_names", []) if ent2 in m_nodes else []
                if _identity_similarity(ent1, alts1, ent2, alts2) >= self.thres_sim:
                    union(ent1, ent2)

        # Merge node definitions
        clustered_nodes = defaultdict(list)
        for ent, val_list in m_nodes.items():
            root = find(ent)
            if ent != root:
                for nd in val_list:
                    alts = nd.get("alternate_names", [])
                    if isinstance(alts, list):
                        if ent not in alts:
                            alts.append(ent)
                    elif isinstance(alts, str):
                        alts = [a.strip() for a in alts.split(",") if a.strip()]
                        if ent not in alts:
                            alts.append(ent)
                        nd["alternate_names"] = alts
            clustered_nodes[root].extend(val_list)

        # Update edges
        clustered_edges = defaultdict(list)
        for (src, tgt), edge_list in m_edges.items():
            r_src, r_tgt = find(src), find(tgt)
            if r_src == r_tgt:
                continue
            for edge in edge_list:
                edge["src_id"] = r_src
                edge["tgt_id"] = r_tgt
            key = tuple(sorted((r_src, r_tgt)))
            clustered_edges[key].extend(edge_list)

        return dict(clustered_nodes), dict(clustered_edges)

    def get_core_entities(self, knowledge_graph_inst) -> List[str]:
        """Retrieves entity names currently in Core KG."""
        if hasattr(knowledge_graph_inst, "graph"):
            return list(knowledge_graph_inst.graph.nodes)
        try:
            return list(knowledge_graph_inst._graph.nodes)
        except Exception:
            return []

    async def _get_core_node_data(self, knowledge_graph_inst, entity_name: str) -> dict:
        try:
            node = await knowledge_graph_inst.get_node(entity_name)
            return node if node else {}
        except Exception:
            return {}

    async def _get_core_neighbors(self, knowledge_graph_inst, entity_name: str) -> Set[str]:
        neighbors = set()
        try:
            raw_edges = await knowledge_graph_inst.get_node_edges(entity_name)
            if raw_edges:
                for src, tgt in raw_edges:
                    neighbors.add(tgt if src == entity_name else src)
        except Exception:
            pass
        return neighbors

    async def _llm_decide_merge(
        self,
        s_ent: str,
        c_ent: str,
        s_node_data: dict,
        c_node_data: dict,
    ) -> bool:
        use_llm_func = self.global_config.get("llm_model_func")
        if not use_llm_func:
            return False

        s_type = s_node_data.get("entity_type", "UNKNOWN")
        c_type = c_node_data.get("entity_type", "UNKNOWN")
        s_desc = s_node_data.get("description", "N/A")
        c_desc = c_node_data.get("description", "N/A")

        prompt = f"""You are an expert historian.
Determine if the following two candidate entities refer to the EXACT SAME historical entity and should be merged into a single node.

Entity 1:
- Name: {s_ent}
- Type: {s_type}
- Description: {s_desc}

Entity 2:
- Name: {c_ent}
- Type: {c_type}
- Description: {c_desc}

Answer with ONLY "YES" or "NO"."""
        try:
            res = await use_llm_func(prompt)
            return bool(res and "YES" in str(res).strip().upper())
        except Exception:
            return False

    async def align(self, m_nodes, m_edges, knowledge_graph_inst):
        # 1. Local clustering within chunk
        m_nodes, m_edges = self._cluster_local_kg(m_nodes, m_edges)

        small_entities = list(m_nodes.keys())
        core_entities = self.get_core_entities(knowledge_graph_inst)

        if not core_entities:
            return m_nodes, m_edges

        name_mapping: Dict[str, str] = {}
        temporal_mappings: List[Tuple[str, str]] = []

        working_dir = self.global_config.get("working_dir", ".")
        log_file_path = os.path.join(working_dir, "fuzzy_alignment_decision_log.txt")

        # 2. Match small entities to core entities
        for s_ent in small_entities:
            s_node_list = m_nodes.get(s_ent, [{}])
            s_node = s_node_list[0] if isinstance(s_node_list, list) and s_node_list else {}
            s_alts = s_node.get("alternate_names", [])
            s_time = s_node.get("time_range")
            s_type = (s_node.get("entity_type") or "").strip().upper()

            # Small KG local neighbors for s_ent
            s_neighbors = set()
            for (src, tgt) in m_edges.keys():
                if src == s_ent:
                    s_neighbors.add(name_mapping.get(tgt, tgt))
                elif tgt == s_ent:
                    s_neighbors.add(name_mapping.get(src, src))

            best_match = None
            best_decision = ""
            is_temporal = False

            for c_ent in core_entities:
                c_node = await self._get_core_node_data(knowledge_graph_inst, c_ent)
                c_type = (c_node.get("entity_type") or "").strip().upper()

                # Entity type safety check
                if s_type and c_type and s_type != "UNKNOWN" and c_type != "UNKNOWN" and s_type != c_type:
                    continue

                c_alts = c_node.get("alternate_names", [])
                c_time = c_node.get("time_range")

                id_sim = _identity_similarity(s_ent, s_alts, c_ent, c_alts)
                t_status = _temporal_status(s_time, c_time)

                c_neighbors = await self._get_core_neighbors(knowledge_graph_inst, c_ent)
                neigh_sim = _neighbourhood_similarity(s_neighbors, c_neighbors)

                # Decision Rules
                should_merge = False
                decision_reason = ""

                if self.use_llm_fuzzy and (id_sim >= self.thres_sim or neigh_sim >= self.thres_sim):
                    should_merge = await self._llm_decide_merge(s_ent, c_ent, s_node, c_node)
                    decision_reason = "LLM decision"
                else:
                    if id_sim >= self.thres_map and t_status == "overlap":
                        should_merge = True
                        decision_reason = f"Identity sim ({id_sim:.2f}) + Time overlap"
                    elif id_sim >= self.thres_map and t_status == "no_info":
                        should_merge = True
                        decision_reason = f"Identity sim ({id_sim:.2f}) + No time info"
                    elif neigh_sim >= self.thres_map:
                        should_merge = True
                        decision_reason = f"Neighbourhood sim ({neigh_sim:.2f})"
                    elif id_sim >= self.thres_map and t_status == "mismatch":
                        is_temporal = True

                if should_merge:
                    best_match = c_ent
                    best_decision = f"MERGED ({decision_reason})"
                    break
                elif is_temporal:
                    best_match = c_ent
                    best_decision = "TEMPORAL MAP"

            if best_match:
                if best_decision.startswith("MERGED"):
                    name_mapping[s_ent] = best_match
                elif best_decision == "TEMPORAL MAP":
                    temporal_mappings.append((s_ent, best_match))

                try:
                    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
                    with open(log_file_path, "a", encoding="utf-8") as f_log:
                        f_log.write(
                            f"Alignment Match:\n"
                            f"  Small Entity: {s_ent} -> Core: {best_match}\n"
                            f"  Decision: {best_decision}\n"
                            f"--------------------------------------------------\n"
                        )
                except Exception:
                    pass

        # 3. Apply mappings to nodes and edges
        aligned_nodes = defaultdict(list)
        for s_ent, node_list in m_nodes.items():
            target_name = name_mapping.get(s_ent, s_ent)
            for node in node_list:
                node["entity_name"] = target_name
                if s_ent != target_name:
                    alts = node.get("alternate_names", [])
                    if isinstance(alts, list):
                        if s_ent not in alts:
                            alts.append(s_ent)
                    elif isinstance(alts, str):
                        alts = [a.strip() for a in alts.split(",") if a.strip()]
                        if s_ent not in alts:
                            alts.append(s_ent)
                        node["alternate_names"] = alts
            aligned_nodes[target_name].extend(node_list)

        aligned_edges = defaultdict(list)
        for (src, tgt), edge_list in m_edges.items():
            new_src = name_mapping.get(src, src)
            new_tgt = name_mapping.get(tgt, tgt)
            for edge in edge_list:
                edge["src_id"] = new_src
                edge["tgt_id"] = new_tgt
            aligned_edges[tuple(sorted((new_src, new_tgt)))].extend(edge_list)

        # Temporal mappings edges
        for s_ent, c_ent in temporal_mappings:
            mapped_src = name_mapping.get(s_ent, s_ent)
            mapped_tgt = name_mapping.get(c_ent, c_ent)
            if mapped_src != mapped_tgt:
                key = tuple(sorted((mapped_src, mapped_tgt)))
                aligned_edges[key].append(
                    {
                        "src_id": mapped_src,
                        "tgt_id": mapped_tgt,
                        "weight": 1.0,
                        "description": f"Temporal mapping between {mapped_src} and {mapped_tgt}.",
                        "keywords": "RENAMED_TO, temporal_mapping",
                        "source_id": "fuzzy_alignment_temporal",
                        "time_range": "unknown",
                    }
                )

        return dict(aligned_nodes), dict(aligned_edges)
