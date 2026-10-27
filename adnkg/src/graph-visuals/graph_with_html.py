"""
Knowledge Graph Visualizer for MiniRAG (NetworkXStorage)
========================================================
Generates an interactive HTML visualization from a .graphml file
produced by MiniRAG's NetworkXStorage backend.

Usage:
    python graph_with_html.py --graph PATH_TO_GRAPHML [--output OUTPUT.html] [--title TITLE]

    # Compare two KGs side-by-side (opens two HTML files)
    python graph_with_html.py --graph path1.graphml --compare path2.graphml
"""

import argparse
import hashlib
import os
import sys

import pipmaster as pm

if not pm.is_installed("pyvis"):
    pm.install("pyvis")
if not pm.is_installed("networkx"):
    pm.install("networkx")

import networkx as nx
from pyvis.network import Network


# ---------------------------------------------------------------------------
# Color palette: map entity_type -> colour hex.  Unknown types get hashed.
# ---------------------------------------------------------------------------
ENTITY_TYPE_COLORS = {
    "PERSON":        "#e74c3c",
    "ORGANIZATION":  "#3498db",
    "LOCATION":      "#2ecc71",
    "EVENT":         "#f39c12",
    "DATE":          "#9b59b6",
    "CONCEPT":       "#1abc9c",
    "OBJECT":        "#e67e22",
    "DOCUMENT":      "#34495e",
    "TECHNOLOGY":    "#00bcd4",
    "UNKNOWN":       "#95a5a6",
}


def _color_for_type(entity_type: str) -> str:
    """Return a deterministic colour for a given entity_type string."""
    cleaned = entity_type.strip('"').strip("'").upper()
    if cleaned in ENTITY_TYPE_COLORS:
        return ENTITY_TYPE_COLORS[cleaned]
    # Deterministic hash-based colour for any other type
    h = int(hashlib.md5(cleaned.encode()).hexdigest()[:6], 16)
    return f"#{h:06x}"


def load_graph(path: str) -> nx.Graph:
    """Load a .graphml or .gml file into a NetworkX graph."""
    if not os.path.exists(path):
        print(f"Error: graph file not found: {path}")
        sys.exit(1)
    if path.endswith(".gml"):
        return nx.read_gml(path)
    else:
        return nx.read_graphml(path)


def build_pyvis(G: nx.Graph, title: str = "Knowledge Graph") -> Network:
    """Convert a NetworkX graph into a styled Pyvis network."""
    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#1a1a2e",       # dark background
        font_color="#eaeaea",
        heading=title,
        notebook=False,
        directed=G.is_directed(),
    )

    # Physics layout options for readability
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.008,
          "springLength": 180,
          "springConstant": 0.04,
          "damping": 0.5
        },
        "solver": "forceAtlas2Based",
        "stabilization": {
          "enabled": true,
          "iterations": 200
        }
      },
      "nodes": {
        "font": { "size": 14, "face": "Inter, sans-serif" },
        "borderWidth": 2,
        "shadow": true
      },
      "edges": {
        "color": { "inherit": "both" },
        "smooth": { "type": "continuous" },
        "font": { "size": 10, "align": "middle", "color": "#aaaaaa" },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      }
    }
    """)

    # --- Add nodes ---
    for node_id, attrs in G.nodes(data=True):
        entity_type = attrs.get("entity_type", "UNKNOWN").strip('"').strip("'")
        description = attrs.get("description", "")
        source_id = attrs.get("source_id", "")
        color = _color_for_type(entity_type)

        # Build rich tooltip
        tooltip_parts = [f"<b>{node_id}</b>"]
        tooltip_parts.append(f"<i>Type:</i> {entity_type}")
        if description:
            # Truncate very long descriptions for tooltip readability
            desc_short = description[:500] + "..." if len(description) > 500 else description
            tooltip_parts.append(f"<i>Description:</i> {desc_short}")
        if source_id:
            src_short = source_id[:200] + "..." if len(source_id) > 200 else source_id
            tooltip_parts.append(f"<i>Sources:</i> {src_short}")
        tooltip = "<br>".join(tooltip_parts)

        # Node size based on degree (computed later, default=15)
        net.add_node(
            node_id,
            label=node_id,
            title=tooltip,
            color=color,
            size=15,
            shape="dot",
        )

    # --- Scale node sizes by degree ---
    degrees = dict(G.degree())
    if degrees:
        max_deg = max(degrees.values()) or 1
        for node in net.nodes:
            deg = degrees.get(node["id"], 1)
            node["size"] = 10 + 30 * (deg / max_deg)

    # --- Add edges ---
    for src, tgt, attrs in G.edges(data=True):
        description = attrs.get("description", "")
        keywords = attrs.get("keywords", "")
        weight = attrs.get("weight", 1)

        tooltip_parts = [f"<b>{src}</b> → <b>{tgt}</b>"]
        if keywords:
            kw_short = keywords[:300] + "..." if len(keywords) > 300 else keywords
            tooltip_parts.append(f"<i>Keywords:</i> {kw_short}")
        if description:
            desc_short = description[:400] + "..." if len(description) > 400 else description
            tooltip_parts.append(f"<i>Description:</i> {desc_short}")
        tooltip = "<br>".join(tooltip_parts)

        # Edge label: first keyword or empty
        label = ""
        if keywords:
            first_kw = keywords.split("<SEP>")[0].strip() if "<SEP>" in keywords else keywords.split(",")[0].strip()
            if len(first_kw) < 40:
                label = first_kw

        try:
            w = float(weight)
        except (ValueError, TypeError):
            w = 1.0

        net.add_edge(
            src,
            tgt,
            title=tooltip,
            label=label,
            value=w,     # pyvis uses 'value' for edge thickness
            width=max(1, min(w, 8)),
        )

    return net


def visualize(graph_path: str, output_path: str, title: str):
    """Load graph, build pyvis, save HTML."""
    G = load_graph(graph_path)
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    net = build_pyvis(G, title=title)
    net.save_graph(output_path)
    print(f"Saved interactive visualization to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize MiniRAG Knowledge Graph (NetworkXStorage .graphml)"
    )
    parser.add_argument(
        "--graph", type=str, required=True,
        help="Path to .graphml file (e.g. output/Vhis-no-fl/graph_chunk_entity_relation.graphml)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML file path (default: <graph_dir>/knowledge_graph.html)"
    )
    parser.add_argument(
        "--title", type=str, default="MiniRAG Knowledge Graph",
        help="Title displayed on the HTML page"
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Path to a second .graphml file for side-by-side comparison"
    )
    args = parser.parse_args()

    # Primary graph
    output1 = args.output or os.path.join(
        os.path.dirname(args.graph), "knowledge_graph.html"
    )
    visualize(args.graph, output1, title=args.title)

    # Optional second graph for comparison
    if args.compare:
        output2 = os.path.join(
            os.path.dirname(args.compare), "knowledge_graph.html"
        )
        title2 = args.title + " (Comparison)"
        visualize(args.compare, output2, title=title2)
        print(f"\n=== Comparison ===")
        G1 = load_graph(args.graph)
        G2 = load_graph(args.compare)
        print(f"Graph 1: {G1.number_of_nodes()} nodes, {G1.number_of_edges()} edges  [{args.graph}]")
        print(f"Graph 2: {G2.number_of_nodes()} nodes, {G2.number_of_edges()} edges  [{args.compare}]")
        node_diff = G1.number_of_nodes() - G2.number_of_nodes()
        edge_diff = G1.number_of_edges() - G2.number_of_edges()
        print(f"Difference: {node_diff:+d} nodes, {edge_diff:+d} edges")


if __name__ == "__main__":
    main()
