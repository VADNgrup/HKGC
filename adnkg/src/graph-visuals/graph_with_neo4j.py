import argparse
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
import networkx as nx


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(REPO_ROOT, ".env"), override=False)

DEFAULT_WORKING_DIR = os.getenv("WORKING_DIR", "./LiHua-World")
DEFAULT_GRAPHML = os.path.join(DEFAULT_WORKING_DIR, "neo4j_graph.graphml")
DEFAULT_ENTITIES = os.path.join(DEFAULT_WORKING_DIR, "entites.txt")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "your_password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")


def get_args():
    parser = argparse.ArgumentParser(description="Export Neo4j KG for visualization")
    parser.add_argument(
        "--graphml",
        type=str,
        default=DEFAULT_GRAPHML,
        help="Output GraphML path",
    )
    parser.add_argument(
        "--entities",
        type=str,
        default=DEFAULT_ENTITIES,
        help="Output entities list path",
    )
    return parser.parse_args()


def _ensure_parent_dir(path: str):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _pick_entity_name(labels, display_name):
    if display_name:
        return str(display_name)
    if labels:
        return str(labels[0])
    return "UNKNOWN"


def export_graph(driver, graphml_path: str, entities_path: str, database: str | None):
    graph = nx.MultiDiGraph()
    entities = set()

    node_query = """
        MATCH (n)
        RETURN id(n) AS nid,
               labels(n) AS labels,
               n.displayName AS displayName,
               n.entity_type AS entity_type,
               n.description AS description,
               n.source_id AS source_id
    """

    edge_query = """
        MATCH (a)-[r]->(b)
        RETURN id(a) AS source_id,
               id(b) AS target_id,
               type(r) AS rel_type,
               properties(r) AS props
    """

    with driver.session(database=database) as session:
        for record in session.run(node_query):
            nid = record["nid"]
            labels = record["labels"] or []
            display_name = record.get("displayName")
            entity_name = _pick_entity_name(labels, display_name)
            entities.add(entity_name)
            graph.add_node(
                nid,
                entity_name=entity_name,
                labels=labels,
                entity_type=record.get("entity_type"),
                description=record.get("description"),
                source_id=record.get("source_id"),
            )

        for record in session.run(edge_query):
            source_id = record["source_id"]
            target_id = record["target_id"]
            rel_type = record.get("rel_type")
            props = record.get("props") or {}
            graph.add_edge(source_id, target_id, key=rel_type, rel_type=rel_type, **props)

    _ensure_parent_dir(graphml_path)
    _ensure_parent_dir(entities_path)

    nx.write_graphml(graph, graphml_path)
    with open(entities_path, "w", encoding="utf-8") as f:
        for name in sorted(entities):
            f.write(f"{name}\n")

    print(f"GraphML saved to: {graphml_path}")
    print(f"Entities list saved to: {entities_path}")
    print(f"Nodes: {graph.number_of_nodes()}, Edges: {graph.number_of_edges()}")


def main():
    args = get_args()
    database = NEO4J_DATABASE or None

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        export_graph(driver, args.graphml, args.entities, database)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
