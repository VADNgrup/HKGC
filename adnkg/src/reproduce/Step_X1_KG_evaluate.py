import os
import sys
import argparse
import networkx as nx
from dotenv import load_dotenv

# Load env variables for Neo4j credentials
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(REPO_ROOT, ".env"), override=False)

def load_normalized_source_texts(datapath):
    import glob
    # Append path to import minirag schema normalizer
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if REPO_ROOT not in sys.path:
        sys.path.append(REPO_ROOT)
    
    try:
        from minirag.schema import _normalizer
    except ImportError:
        # Simple fallback character removal
        class FallbackNormalizer:
            def remove_diacritics(self, text):
                import re
                text = re.sub(r'[àáạảãâầấậẩẫăằắặẳẵ]', 'a', text)
                text = re.sub(r'[ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]', 'A', text)
                text = re.sub(r'[èéẹẻẽêềếệểễ]', 'e', text)
                text = re.sub(r'[ÈÉẸẺẼÊỀẾỆỂỄ]', 'E', text)
                text = re.sub(r'[òóọỏõôồốộổỗơờớợởỡ]', 'o', text)
                text = re.sub(r'[ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ]', 'O', text)
                text = re.sub(r'[ìíịỉĩ]', 'i', text)
                text = re.sub(r'[ÌÍỊỈĨ]', 'I', text)
                text = re.sub(r'[ùúụủũưừứựửữ]', 'u', text)
                text = re.sub(r'[ÙÚỤỦŨƯỪỨỰỬỮ]', 'U', text)
                text = re.sub(r'[ỳýỵỷỹ]', 'y', text)
                text = re.sub(r'[ỲÝỴỶỸ]', 'Y', text)
                text = re.sub(r'[đ]', 'd', text)
                text = re.sub(r'[Đ]', 'D', text)
                return text
        _normalizer = FallbackNormalizer()

    if not datapath or not os.path.exists(datapath):
        return ""

    txt_files = []
    if os.path.isdir(datapath):
        txt_files = glob.glob(os.path.join(datapath, "**/*.txt"), recursive=True)
    elif os.path.isfile(datapath):
        txt_files = [datapath]

    if not txt_files:
        return ""

    print(f"Reading and normalizing {len(txt_files)} source files from {datapath}...")
    normalized_contents = []
    for fpath in txt_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                norm_content = _normalizer.remove_diacritics(content).upper()
                normalized_contents.append(norm_content)
        except Exception as e:
            print(f"Warning: Failed to read source file {fpath}: {e}")

    return "\n".join(normalized_contents)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Knowledge Graph created by MiniRAG")
    parser.add_argument("--workingdir", type=str, default=None, help="Path to the working directory containing the GML graph file (for NetworkX)")
    parser.add_argument("--neo4j", action="store_true", help="Load graph directly from Neo4j database")
    parser.add_argument("--datapath", type=str, default=os.getenv("DATA_PATH", "./dataset/VHis/data"), help="Path to raw source text documents")
    args = parser.parse_args()

    sorted_nodes = []
    nodes_count = 0
    edges_count = 0
    loaded = False
    G = nx.Graph()

    if args.neo4j or (not args.workingdir and os.getenv("GRAPH_STORAGE") == "Neo4JStorage"):
        # Load from Neo4j
        uri = os.getenv("NEO4J_URI", "bolt://localhost:17687")
        user = os.getenv("NEO4J_USERNAME", os.getenv("NEO4J_USER", "neo4j"))
        password = os.getenv("NEO4J_PASSWORD", "password123")
        database = os.getenv("NEO4J_DATABASE", None)
        if database == "":
            database = None

        print(f"Connecting to Neo4j database at {uri}...")
        try:
            from neo4j import GraphDatabase
        except ImportError:
            print("neo4j package not installed. Installing pipmaster and neo4j...")
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "neo4j"])
            from neo4j import GraphDatabase

        try:
            with GraphDatabase.driver(uri, auth=(user, password)) as driver:
                with driver.session(database=database) as session:
                    # Retrieve nodes
                    nodes_result = session.run("MATCH (n) RETURN n.id AS id, labels(n) AS labels, n.entity_type AS type")
                    nodes = []
                    for record in nodes_result:
                        node_id = record.get("id")
                        if node_id:
                            nodes.append(node_id)
                            G.add_node(node_id)
                    sorted_nodes = sorted(list(set(nodes)))
                    nodes_count = len(sorted_nodes)

                    # Retrieve all relationships and add to G
                    edges_result = session.run("MATCH (n)-[r]->(m) RETURN n.id AS src, m.id AS tgt, r.keywords AS keywords, r.description AS description")
                    for record in edges_result:
                        src = record.get("src")
                        tgt = record.get("tgt")
                        if src and tgt:
                            G.add_edge(src, tgt, keywords=record.get("keywords"), description=record.get("description"))

                    edges_count = G.number_of_edges()
                    loaded = True
        except Exception as e:
            print(f"Error loading from Neo4j: {e}")
            print("Falling back to local graph files check...")

    if not loaded:
        if not args.workingdir:
            args.workingdir = os.getenv("WORKING_DIR", "../output/LiHua-World")
            
        graph_file = os.path.join(args.workingdir, "graph_chunk_entity_relation.graphml")
        if not os.path.exists(graph_file):
            graph_file_alt = os.path.join(args.workingdir, "chunk_entity_relation.gml")
            if os.path.exists(graph_file_alt):
                graph_file = graph_file_alt
            else:
                found = False
                if os.path.exists(args.workingdir):
                    for f in os.listdir(args.workingdir):
                        if f.endswith(".graphml") or f.endswith(".gml"):
                            graph_file = os.path.join(args.workingdir, f)
                            found = True
                            break
                if not found:
                    print(f"Error: No graph files found in {args.workingdir} and Neo4j connection failed.")
                    sys.exit(1)

        print(f"Loading Graph from: {graph_file}")
        if graph_file.endswith(".gml"):
            G = nx.read_gml(graph_file)
        else:
            G = nx.read_graphml(graph_file)

        nodes_count = G.number_of_nodes()
        edges_count = G.number_of_edges()
        sorted_nodes = sorted(list(G.nodes()))

    # Read and normalize original source text files for validation
    source_text = load_normalized_source_texts(args.datapath)

    print("\n" + "="*50)
    print("KNOWLEDGE GRAPH EVALUATION REPORT")
    print("="*50)
    print(f"Total Unique Nodes (Entities):     {nodes_count}")
    print(f"Total Unique Edges (Relations):   {edges_count}")
    print("="*50)

    # Sort and list nodes alphabetically
    print("\nNodes List (A-Z):")
    print("-" * 20)
    for node in sorted_nodes:
        print(f"- {node}")

    # Evaluate potential duplicates using Levenshtein similarity
    print("\n" + "="*50)
    print("POTENTIAL DUPLICATE ENTITIES EVALUATION (MAX 25 DETAILS)")
    print("="*50)
    
    import difflib
    duplicates_found = 0
    
    for i in range(len(sorted_nodes)):
        node_i = sorted_nodes[i]
        for j in range(i + 1, len(sorted_nodes)):
            node_j = sorted_nodes[j]
            # Skip short words
            if len(node_i) < 4 or len(node_j) < 4:
                continue
            
            sim_ratio = difflib.SequenceMatcher(None, node_i.lower(), node_j.lower()).ratio()
            if sim_ratio > 0.8:
                duplicates_found += 1
                
                # Only print detailed analysis for the first 25 duplicate candidates
                if duplicates_found <= 25:
                    # 1. Neighbor consistency (Jaccard)
                    neighbors_i = set(G.neighbors(node_i)) if G.has_node(node_i) else set()
                    neighbors_j = set(G.neighbors(node_j)) if G.has_node(node_j) else set()
                    intersection = neighbors_i.intersection(neighbors_j)
                    union = neighbors_i.union(neighbors_j)
                    neighbor_jaccard = len(intersection) / len(union) if union else 0.0
                    
                    # 2. Fact existence directly between them
                    direct_edge = "No"
                    edge_details = ""
                    if G.has_edge(node_i, node_j):
                        direct_edge = "Yes"
                        data = G.get_edge_data(node_i, node_j)
                        # Handle multigraph or simple graph
                        if isinstance(data, dict):
                            keywords = data.get("keywords") or data.get("weight") or ""
                            edge_details = f" (Details: {keywords})"
                    
                    # 3. Source text existence check
                    clean_i = node_i.split("(")[0].strip()
                    clean_j = node_j.split("(")[0].strip()
                    in_source_i = "Yes" if (source_text and clean_i in source_text) else ("No" if source_text else "N/A")
                    in_source_j = "Yes" if (source_text and clean_j in source_text) else ("No" if source_text else "N/A")
                    
                    print(f"[{duplicates_found}] Potential Duplicate: Score: {sim_ratio:.2f} | \"{node_i}\" <==> \"{node_j}\"")
                    print(f"    - Neighbor Score (Jaccard): {neighbor_jaccard:.4f} (Intersection: {len(intersection)}, Union: {len(union)})")
                    if intersection:
                        print(f"      Common Neighbors: {list(intersection)}")
                    print(f"    - Direct edge connecting them: {direct_edge}{edge_details}")
                    print(f"    - Found in raw source text:")
                    print(f"      * \"{node_i}\" -> {in_source_i}")
                    print(f"      * \"{node_j}\" -> {in_source_j}")
                    print("-" * 50)
                
    if duplicates_found == 0:
        print("No potential duplicates detected (Threshold: > 0.80 name similarity).")
    else:
        print(f"Total detected potential duplicate entity pairs: {duplicates_found}")
        if duplicates_found > 25:
            print("  (Detailed statistics truncated after 25 examples.)")
    print("="*50)

if __name__ == "__main__":
    main()
