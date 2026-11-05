import torch
from torch import Tensor
from collections import deque
from typing import List, Tuple, Optional, Dict, Any
from torch_geometric.data import Data

# helper: BFS distances on the induced subgraph
def bfs_dist(sub_edge_index: Tensor, sub_nodes: Tensor, root_local_idx: int, max_k: int):
    # build adjacency list (local indexing: [0..S-1])
    S = sub_nodes.numel()
    adj = [[] for _ in range(S)]
    u_local = sub_edge_index[0].tolist()
    v_local = sub_edge_index[1].tolist()
    for u, v in zip(u_local, v_local):
        adj[u].append(v)
        adj[v].append(u)

    INF = 10**9
    dist = [INF] * S
    q = deque([root_local_idx])
    dist[root_local_idx] = 0
    while q:
        x = q.popleft()
        if dist[x] == max_k:
            continue
        for y in adj[x]:
            if dist[y] == INF:
                dist[y] = dist[x] + 1
                q.append(y)
    # clamp >k to k+ (we’ll represent anything >k as k for onehot/ scalar)
    for i in range(S):
        if dist[i] == INF:
            dist[i] = max_k + 1  # disconnected in induced subgraph (shouldn't happen in k-hop ego)
    return torch.tensor(dist, dtype=torch.long)



def khop_expand_nodes(
    data: Data,
    core_nodes: torch.Tensor,
    k: int,
    return_depth: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    k-hop BFS expansion from 'core_nodes' on the original graph.

    Args:
        data: PyG Data object with edge_index (assumed undirected or symmetric).
        core_nodes: 1D LongTensor of starting nodes.
        k: number of hops (k=0 returns core_nodes only).
        return_depth: if True, also returns BFS distance from nearest core.

    Returns:
        expanded_nodes: 1D LongTensor of nodes with distance <= k.
        depths: 1D LongTensor of same length as expanded_nodes (BFS depth),
                or None if return_depth=False.
    """
    edge_index = data.edge_index
    device = edge_index.device
    N = data.num_nodes
    u, v = edge_index

    core_nodes = core_nodes.to(device)
    if core_nodes.numel() == 0:
        depths = torch.empty_like(core_nodes) if return_depth else None
        return core_nodes, depths

    depth = torch.full((N,), -1, dtype=torch.long, device=device)
    depth[core_nodes] = 0

    mask = torch.zeros(N, dtype=torch.bool, device=device)
    mask[core_nodes] = True

    frontier = core_nodes

    if k == 0:
        expanded_nodes = core_nodes.unique()
        depths = depth[expanded_nodes] if return_depth else None
        return expanded_nodes, depths

    for d in range(1, k + 1):
        if frontier.numel() == 0:
            break

        fmask = torch.zeros(N, dtype=torch.bool, device=device)
        fmask[frontier] = True

        incident = fmask[u] | fmask[v]
        if not incident.any():
            break

        neigh = torch.cat([u[incident], v[incident]], dim=0).unique()
        new_neigh = neigh[depth[neigh] == -1]
        if new_neigh.numel() == 0:
            break

        depth[new_neigh] = d
        mask[new_neigh] = True
        frontier = new_neigh

    expanded_nodes = torch.nonzero(mask, as_tuple=False).view(-1)
    depths = depth[expanded_nodes] if return_depth else None
    return expanded_nodes, depths


def induced_edges_from_nodes(
    data: Data,
    nodes: torch.Tensor,
) -> torch.Tensor:
    """
    Induced edge set on 'nodes'.

    Args:
        data: PyG Data object with edge_index.
        nodes: 1D LongTensor of node indices.

    Returns:
        edge_idx: 1D LongTensor of indices into data.edge_index.
    """
    edge_index = data.edge_index
    device = edge_index.device
    N = data.num_nodes
    u, v = edge_index

    node_mask = torch.zeros(N, dtype=torch.bool, device=device)
    node_mask[nodes] = True

    edge_mask = node_mask[u] & node_mask[v]
    edge_idx = torch.nonzero(edge_mask, as_tuple=False).view(-1)
    return edge_idx


def build_adj_lists(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """
    Build adjacency lists (undirected) from edge_index.

    Args:
        edge_index: LongTensor [2, E]
        num_nodes: N

    Returns:
        adj: list of length N, each entry is a list of neighbor node indices.
    """
    u, v = edge_index
    adj = [[] for _ in range(num_nodes)]
    E = edge_index.size(1)
    for e in range(E):
        a = int(u[e].item())
        b = int(v[e].item())
        adj[a].append(b)
        adj[b].append(a)
    return adj



def _build_marked_subgraph(
    data: Data,
    core_nodes: torch.Tensor,  # global node ids on device
    khop: int,
    use_node_markings: bool,
    use_edge_markings: bool,
) -> Optional[Dict[str, Any]]:
    """
    Given core nodes (global ids), do k-hop expansion, induced edges,
    and attach core/halo markings in exactly the same format as leiden_khop_policy_v2.
    """
    edge_index = data.edge_index
    device = edge_index.device
    N = data.num_nodes

    if core_nodes.numel() == 0:
        return None

    node_ids, depths = khop_expand_nodes(data, core_nodes, k=khop, return_depth=True)
    edge_ids = induced_edges_from_nodes(data, node_ids)
    if node_ids.numel() == 0 or edge_ids.numel() == 0:
        return None

    sg: Dict[str, Any] = {
        "nodes": node_ids,
        "edges": edge_ids,
        "node_markings": None,
        "edge_markings": None,
    }

    # core/halo masks
    core_mask_global = torch.zeros(N, dtype=torch.bool, device=device)
    core_mask_global[core_nodes] = True

    core_flags = core_mask_global[node_ids].float()
    halo_flags = (~core_mask_global[node_ids]).float()

    if use_node_markings:
        node_markings = torch.stack([core_flags, halo_flags], dim=-1)
        sg["node_markings"] = node_markings

    sg["node_marking_dict"] = {
        "is_core": core_flags,
        "is_halo": halo_flags,          # inverse of core
        "depth_core": depths,           # scalar depth from the core set
    }
    sg["is_core"] = core_flags
    sg["is_root"] = None

    if use_edge_markings:
        u, v = edge_index
        eu = u[edge_ids]
        ev = v[edge_ids]

        cu = core_mask_global[eu]
        cv = core_mask_global[ev]

        core_edge = (cu & cv).float()
        cross_edge = (cu ^ cv).float()
        edge_markings = torch.stack([core_edge, cross_edge], dim=-1)
        sg["edge_markings"] = edge_markings

    return sg

