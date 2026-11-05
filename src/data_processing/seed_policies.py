from typing import Any, Dict, List

import torch
from torch_geometric.utils import k_hop_subgraph

from .acl_policy import seeded_tesselating_egonet
from .subgraph_utils import bfs_dist


@torch.no_grad()
def ego_khop_policy(
    data,
    khop: int = 2,
    seeds: torch.Tensor | None = None,
    use_node_markings: bool = True,
    use_edge_markings: bool = False,
):
    """Generate one k-hop ego-subgraph per seed node."""
    device = data.edge_index.device
    N = data.num_nodes
    if seeds is None:
        seeds = torch.arange(N, dtype=torch.long, device=device)
    else:
        seeds = seeds.to(device)

    node_lists, edge_lists, node_markings = [], [], []
    raw_subgraphs: List[Dict[str, Any]] = []
    edge_markings = [] if use_edge_markings else None

    for root in seeds.tolist():
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            root, khop, data.edge_index, relabel_nodes=True, num_nodes=N, flow='source_to_target'
        )
        nodes_i = subset.to(device)
        edges_i = torch.nonzero(edge_mask, as_tuple=False).view(-1).to(device)

        d_local = bfs_dist(sub_edge_index, nodes_i, int(mapping.item()), khop).to(device)

        root_flag = torch.zeros_like(d_local, dtype=torch.float)
        root_flag[mapping] = 1.0

        d_clamped = torch.clamp(d_local, max=khop)
        marks_i = torch.stack([root_flag, d_clamped], dim=-1)

        node_marking_dict = {
            "is_root": root_flag,
            "depth_root": d_clamped,
        }

        node_lists.append(nodes_i)
        edge_lists.append(edges_i)
        node_markings.append(marks_i)

        edge_mark = None
        if use_edge_markings:
            local_of = {int(n.item()): i for i, n in enumerate(nodes_i)}
            src = data.edge_index[0, edges_i].tolist()
            dst = data.edge_index[1, edges_i].tolist()

            same_shell = []
            cross_shell = []
            touches_root = []
            for u, v in zip(src, dst):
                du = int(d_local[local_of[u]].item())
                dv = int(d_local[local_of[v]].item())
                same_shell.append(1.0 if du == dv else 0.0)
                cross_shell.append(1.0 if du != dv else 0.0)
                touches_root.append(1.0 if (du == 0 or dv == 0) else 0.0)

            edge_mark = torch.tensor(
                list(zip(same_shell, cross_shell, touches_root)),
                dtype=torch.float, device=device
            )
            edge_markings.append(edge_mark)

        raw_subgraphs.append({
            "nodes": nodes_i,
            "edges": edges_i,
            "node_markings": marks_i,
            "edge_markings": edge_mark,
            "node_marking_dict": node_marking_dict,
            "is_root": root_flag,
            "is_core": None,
        })

    return raw_subgraphs


def whole_graph_policy(data):
    N = data.num_nodes
    E = data.edge_index.size(1)

    node_lists = [torch.arange(N, dtype=torch.long)]
    edge_lists = [torch.arange(E, dtype=torch.long)]

    node_markings = None

    marking_dict = {
    }
    raw_subgraphs = [
        {
            "nodes": node_lists[0],
            "edges": edge_lists[0],
            "node_markings": node_markings,
            "node_marking_dict": marking_dict,
        }
    ]
    return raw_subgraphs
