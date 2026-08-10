#!/usr/bin/env python3
import argparse
import os
import sys
import difflib
import textwrap
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def load_graph(path: str) -> nx.Graph:
    if not os.path.exists(path):
        print(f"Error: graph file not found: {path}")
        sys.exit(1)
    if path.endswith(".gml"):
        return nx.read_gml(path)
    else:
        return nx.read_graphml(path)

def find_subgraph(G: nx.Graph, center_node: str, radius: int = 1):
    nodes_in_g = [n for n in G.nodes() if center_node.upper() in n.upper()]
    if not nodes_in_g:
        return None, None
    exact = [n for n in nodes_in_g if n.upper() == center_node.upper()]
    actual_center = exact[0] if exact else nodes_in_g[0]
    
    sub_nodes = set([actual_center])
    current_frontier = set([actual_center])
    for _ in range(radius):
        next_frontier = set()
        for n in current_frontier:
            next_frontier.update(G.neighbors(n))
        sub_nodes.update(next_frontier)
        current_frontier = next_frontier
    return actual_center, G.subgraph(sub_nodes).copy()

def find_similar_nodes(G1: nx.Graph, G2: nx.Graph, target_node: str, top_k: int = 5):
    nodes2 = list(G2.nodes())
    matches = difflib.get_close_matches(target_node, nodes2, n=top_k, cutoff=0.2)
    results = []
    for m in matches:
        score = difflib.SequenceMatcher(None, target_node.upper(), m.upper()).ratio()
        results.append((m, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def truncate_field(val: str, max_len: int = 70) -> str:
    if not val:
        return ""
    val_str = str(val).strip()
    if len(val_str) > max_len:
        return val_str[:max_len - 3] + "..."
    return val_str

def format_node_schema_text(node_name: str, G: nx.Graph, keyword: str) -> str:
    if node_name not in G.nodes():
        return f"• Status: Node matching '{keyword}' not present in graph."
    
    attrs = G.nodes[node_name]
    lines = [f"• Entity Name: {node_name}"]
    
    ent_type = attrs.get("entity_type", "UNKNOWN")
    lines.append(f"• Entity Type: {truncate_field(ent_type, 65)}")
    
    alts = attrs.get("alternate_names", "")
    if isinstance(alts, list):
        alts_str = ", ".join(alts)
    else:
        alts_str = str(alts)
    alts_clean = alts_str.replace("<SEP>", " | ")
    if alts_clean and alts_clean.lower() != "none":
        lines.append(f"• Alternate Names: {truncate_field(alts_clean, 70)}")
        
    time_range = str(attrs.get("time_range", "")).replace("<SEP>", " | ")
    if time_range and time_range.lower() != "none" and time_range.lower() != "unknown":
        lines.append(f"• Time Range: {truncate_field(time_range, 70)}")
        
    geo = str(attrs.get("geo_location", "")).replace("<SEP>", " | ")
    if geo and geo.lower() != "none" and geo.lower() != "unknown":
        lines.append(f"• Geo Location: {truncate_field(geo, 70)}")
        
    lines.append(f"• Graph Degree: {G.degree(node_name)}")
    
    desc = attrs.get("description", "No description available.")
    desc_clean = str(desc).replace("<SEP>", " | ")
    
    lines.append("• Description:")
    wrapped_lines = textwrap.wrap(desc_clean, width=65)
    max_desc_lines = 3
    for i, line in enumerate(wrapped_lines[:max_desc_lines]):
        if i == max_desc_lines - 1 and len(wrapped_lines) > max_desc_lines:
            line_trunc = truncate_field(line, 62)
            if not line_trunc.endswith("..."):
                line_trunc += "..."
            lines.append(f"   {line_trunc}")
        else:
            lines.append(f"   {line}")
            
    if len(wrapped_lines) > max_desc_lines and not lines[-1].endswith("..."):
        lines.append("   ...")
        
    return "\n".join(lines)

def plot_comparison(graph1_path: str, graph2_path: str, node_name: str, output_img_path: str, title1: str = "KG 1", title2: str = "KG 2", radius: int = 1):
    G1 = load_graph(graph1_path)
    G2 = load_graph(graph2_path)

    center1, sub1 = find_subgraph(G1, node_name, radius=radius)
    center2, sub2 = find_subgraph(G2, node_name, radius=radius)

    if center1 is None and center2 is None:
        print(f"Warning: Node '{node_name}' not found in either graph. Using top degree node from G1.")
        top_nodes1 = sorted(G1.degree(), key=lambda x: x[1], reverse=True)
        if top_nodes1:
            node_name = top_nodes1[0][0]
            center1, sub1 = find_subgraph(G1, node_name, radius=radius)
            center2, sub2 = find_subgraph(G2, node_name, radius=radius)

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
    fig = plt.figure(figsize=(16, 11), facecolor='white')
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.35, wspace=0.15)
    fig.suptitle(f"Knowledge Graph Subgraph & Schema Comparison: '{node_name}'", fontsize=15, color='#111111', fontweight='bold', y=0.98)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.set_facecolor('white')
    n_cnt1 = sub1.number_of_nodes() if sub1 else 0
    e_cnt1 = sub1.number_of_edges() if sub1 else 0
    ax1.set_title(f"(a) {title1}\n[{n_cnt1} nodes, {e_cnt1} edges]", color='#222222', fontsize=10, fontweight='bold', pad=6)

    if sub1 and n_cnt1 > 0:
        pos1 = nx.spring_layout(sub1, k=0.8, seed=42, iterations=100)
        colors1 = ['#d9534f' if n == center1 else '#5bc0de' for n in sub1.nodes()]
        sizes1 = [800 if n == center1 else 400 for n in sub1.nodes()]
        nx.draw_networkx_nodes(sub1, pos1, ax=ax1, node_color=colors1, node_size=sizes1, edgecolors='#333333', linewidths=1.2, alpha=0.95)
        nx.draw_networkx_edges(sub1, pos1, ax=ax1, edge_color='#888888', width=1.2, alpha=0.7)
        labels1 = {n: (n[:16] + '...') if len(n) > 19 else n for n in sub1.nodes()}
        nx.draw_networkx_labels(sub1, pos1, labels=labels1, ax=ax1, font_color='#111111', font_size=7.5, font_weight='bold')
    else:
        ax1.text(0.5, 0.5, f"Node matching '{node_name}'\nnot present in KG 1", color='#d9534f', fontsize=11, ha='center', va='center', fontweight='bold')
    ax1.axis('off')

    ax2.set_facecolor('white')
    n_cnt2 = sub2.number_of_nodes() if sub2 else 0
    e_cnt2 = sub2.number_of_edges() if sub2 else 0
    ax2.set_title(f"(b) {title2}\n[{n_cnt2} nodes, {e_cnt2} edges]", color='#222222', fontsize=10, fontweight='bold', pad=6)

    if sub2 and n_cnt2 > 0:
        pos2 = nx.spring_layout(sub2, k=0.8, seed=42, iterations=100)
        colors2 = ['#d9534f' if n == center2 else '#5cb85c' for n in sub2.nodes()]
        sizes2 = [800 if n == center2 else 400 for n in sub2.nodes()]
        nx.draw_networkx_nodes(sub2, pos2, ax=ax2, node_color=colors2, node_size=sizes2, edgecolors='#333333', linewidths=1.2, alpha=0.95)
        nx.draw_networkx_edges(sub2, pos2, ax=ax2, edge_color='#888888', width=1.2, alpha=0.7)
        labels2 = {n: (n[:16] + '...') if len(n) > 19 else n for n in sub2.nodes()}
        nx.draw_networkx_labels(sub2, pos2, labels=labels2, ax=ax2, font_color='#111111', font_size=7.5, font_weight='bold')
    else:
        ax2.text(0.5, 0.5, f"Node matching '{node_name}'\nnot present in KG 2", color='#d9534f', fontsize=11, ha='center', va='center', fontweight='bold')
    ax2.axis('off')

    ax3.set_facecolor('white')
    ax3.set_title(f"(c) Schema & Attributes: {title1}", color='#222222', fontsize=10, fontweight='bold', loc='left', pad=6)
    ax3.axis('off')
    
    if center1:
        info1_text = format_node_schema_text(center1, G1, node_name)
        sim1 = find_similar_nodes(G1, G1, center1, top_k=4)
        sim1_str = ", ".join([f"{n} ({s:.2f})" for n, s in sim1 if n != center1])
        if sim1_str:
            info1_text += f"\n• Similar Entities in KG1: {truncate_field(sim1_str, 65)}"
    else:
        info1_text = f"• Status: Entity '{node_name}' not present in {title1}"
        sim_cross1 = find_similar_nodes(G1, G1, node_name, top_k=3)
        if sim_cross1:
            info1_text += "\n• Closest Matches: " + truncate_field(", ".join([f"{n} ({s:.2f})" for n, s in sim_cross1]), 65)
            
    ax3.text(0.01, 0.95, info1_text, color='#222222', fontsize=8, va='top', ha='left', transform=ax3.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#cccccc", lw=1))

    ax4.set_facecolor('white')
    ax4.set_title(f"(d) Schema & Attributes: {title2}", color='#222222', fontsize=10, fontweight='bold', loc='left', pad=6)
    ax4.axis('off')

    if center2:
        info2_text = format_node_schema_text(center2, G2, node_name)
        sim2 = find_similar_nodes(G2, G2, center2, top_k=4)
        sim2_str = ", ".join([f"{n} ({s:.2f})" for n, s in sim2 if n != center2])
        if sim2_str:
            info2_text += f"\n• Similar Entities in KG2: {truncate_field(sim2_str, 65)}"
    else:
        info2_text = f"• Status: Entity '{node_name}' not present in {title2}"
        sim_cross2 = find_similar_nodes(G1, G2, node_name, top_k=3)
        if sim_cross2:
            info2_text += "\n• Closest Matches in KG2: " + truncate_field(", ".join([f"{n} ({s:.2f})" for n, s in sim_cross2]), 65)
            
    ax4.text(0.01, 0.95, info2_text, color='#222222', fontsize=8, va='top', ha='left', transform=ax4.transAxes,
             bbox=dict(boxstyle="round,pad=0.5", fc="#f8f9fa", ec="#cccccc", lw=1))

    target_patch = mpatches.Patch(color='#d9534f', label='Target Keyword Node')
    neighbor1_patch = mpatches.Patch(color='#5bc0de', label='KG1 Neighbor Nodes')
    neighbor2_patch = mpatches.Patch(color='#5cb85c', label='KG2 Neighbor Nodes')
    fig.legend(handles=[target_patch, neighbor1_patch, neighbor2_patch], loc='lower center', ncol=3, frameon=True, facecolor='#ffffff', edgecolor='#cccccc', fontsize=9.5)

    os.makedirs(os.path.dirname(os.path.abspath(output_img_path)), exist_ok=True)
    plt.savefig(output_img_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Publication-ready comparison graph saved to: {output_img_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare 2 Knowledge Graphs and visualize target node & similar entities for paper inclusion")
    parser.add_argument("--dir1", type=str, required=True, help="Directory of first KG output")
    parser.add_argument("--dir2", type=str, required=True, help="Directory of second KG output")
    parser.add_argument("--nodes", type=str, nargs='+', default=["Việt Minh"], help="One or multiple target node names")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory to save images for each node")
    parser.add_argument("--title1", type=str, default="KG 1 (Without Schema / Default)", help="Title for KG 1")
    parser.add_argument("--title2", type=str, default="KG 2 (With FL)", help="Title for KG 2")
    parser.add_argument("--radius", type=int, default=1, help="Hop distance for subgraph visualization (1 or 2)")

    args = parser.parse_args()

    g1_path = os.path.join(args.dir1, "graph_chunk_entity_relation.graphml")
    g2_path = os.path.join(args.dir2, "graph_chunk_entity_relation.graphml")

    output_dir = args.output_dir if args.output_dir else args.dir1
    os.makedirs(output_dir, exist_ok=True)

    for i, node_name in enumerate(args.nodes, 1):
        out_filename = f"comparekg_node_{i}.png"
        output_img_path = os.path.join(output_dir, out_filename)

        print(f"\n[{i}/{len(args.nodes)}] Processing visualization for node: '{node_name}' -> {output_img_path}")
        plot_comparison(
            graph1_path=g1_path,
            graph2_path=g2_path,
            node_name=node_name,
            output_img_path=output_img_path,
            title1=args.title1,
            title2=args.title2,
            radius=args.radius,
        )

if __name__ == "__main__":
    main()
