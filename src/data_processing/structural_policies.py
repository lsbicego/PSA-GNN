import igraph as ig
import leidenalg
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce, remove_self_loops
from collections import defaultdict

from typing import List, Dict, Any, Optional, Tuple, Union, Literal

from .subgraph_utils import khop_expand_nodes, induced_edges_from_nodes

# =============================================================================
# LEIDEN NODE PARTITION + K-HOPS (NON-OVERLAPPING)
# =============================================================================

def leiden_khop_policy_v2(
    data: Data,
    resolution: float = 1.0,
    seed: int = 0,
    khop: int = 1,
    use_node_markings: bool = True,
    use_edge_markings: bool = True,
) -> Dict[str, Optional[List]]:
    """Build overlapping subgraphs from Leiden communities plus k-hop expansion.

    Args:
        data: Input graph.
        resolution: Leiden resolution parameter.
        seed: Random seed for the community detection step.
        khop: Expansion radius around each community core.
        use_node_markings: Whether to attach node-level subgraph markings.
        use_edge_markings: Whether to attach edge-level subgraph markings.

    Returns:
        A list of raw subgraph dictionaries describing expanded Leiden-based
        subgraphs.
    """
    edge_index = data.edge_index
    device = edge_index.device
    N = data.num_nodes

    edges = edge_index.t().cpu().numpy().tolist()
    g = ig.Graph(n=N, edges=edges, directed=False)

    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )

    communities = [
        torch.tensor(comm, dtype=torch.long, device=device) for comm in partition
    ]

    raw_subgraphs: List[Dict[str, Any]] = []

    for comm_nodes in communities:
        if comm_nodes.numel() == 0:
            continue

        node_ids, depths = khop_expand_nodes(data, comm_nodes, k=khop, return_depth=True)
        edge_ids = induced_edges_from_nodes(data, node_ids)
        if node_ids.numel() == 0 or edge_ids.numel() == 0:
            continue

        sg: Dict[str, Any] = {
            "nodes": node_ids,
            "edges": edge_ids,
            "node_markings": None,
            "edge_markings": None,
        }

        core_mask_global = torch.zeros(N, dtype=torch.bool, device=device)
        core_mask_global[comm_nodes] = True

        core_flags = core_mask_global[node_ids].float()
        halo_flags = (~core_mask_global[node_ids]).float()
        node_markings = torch.stack([core_flags, halo_flags], dim=-1)
        sg["node_markings"] = node_markings

        sg["node_marking_dict"] = {
            "is_core": core_flags,
            "is_halo": halo_flags, # just the inverse of is_core
            "depth_core": depths, # scalar depth from core nodes
        }
        sg["is_core"] = core_flags
        sg["is_root"] = None

        if use_edge_markings:
            u, v = edge_index
            eu = u[edge_ids]
            ev = v[edge_ids]

            core_mask_global = torch.zeros(N, dtype=torch.bool, device=device)
            core_mask_global[comm_nodes] = True
            cu = core_mask_global[eu]
            cv = core_mask_global[ev]

            core_edge = (cu & cv).float()
            cross_edge = (cu ^ cv).float()
            edge_markings = torch.stack([core_edge, cross_edge], dim=-1)
            sg["edge_markings"] = edge_markings

        raw_subgraphs.append(sg)

    return raw_subgraphs


# =============================================================================
# OVERLAPPING LEIDEN WITH NODE DUPLICATION
# =============================================================================
# NOTE: This older Leiden duplication policy is currently unused in the
# active SubgraphTransform construction path.
def overlapping_leiden_policy(
    data: Data,
    resolution: float = 1.0,
    seed: int = 0,
    khop: int = 1,
    tau: float = 0.5,
    min_edges: int = 2,
    use_node_markings: bool = True,
    use_edge_markings: bool = True,
) -> Dict[str, Optional[List]]:
    """
    Overlapping communities via Leiden + heuristic node duplication:

      1. Run Leiden -> get hard partition comm_id[v].
      2. For each node v:
           - primary = comm_id[v]
           - count edges from v into each community
           - for any other community c:
                 if edges_to_c >= min_edges and edges_to_c >= tau * edges_to_primary:
                     mark v as secondary member of c as well.
      3. For each community k:
           - core nodes = primary ∪ secondary members
           - k-hop expand to get halo
           - build subgraph & markings.

    Markings:
      - node_markings: [num_sub_nodes, 3]
          [:,0] = 1 if node is primary member of community k
          [:,1] = 1 if node is secondary (duplicated) member of community k
          [:,2] = 1 if halo (in k-hop but not core)
      - edge_markings: [num_sub_edges, 3]
          [:,0] = 1 if both endpoints primary
          [:,1] = 1 if at least one endpoint secondary
          [:,2] = 1 if both endpoints halo
    """
    edge_index = data.edge_index
    device = edge_index.device
    N = data.num_nodes
    u, v = edge_index
    E = edge_index.size(1)

    edges_np = edge_index.t().cpu().numpy().tolist()
    g = ig.Graph(n=N, edges=edges_np, directed=False)
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=resolution,
        seed=seed,
    )
    comm_id = torch.tensor(partition.membership, dtype=torch.long, device=device)
    num_comms = int(comm_id.max().item()) + 1

    # counts[v, c] = number of neighbors of v in community c
    counts = torch.zeros(N, num_comms, dtype=torch.long, device=device)
    for e_idx in range(E):
        a = u[e_idx]
        b = v[e_idx]
        ca = comm_id[a]
        cb = comm_id[b]
        counts[a, cb] += 1
        counts[b, ca] += 1

    primary_members: List[List[int]] = [[] for _ in range(num_comms)]
    secondary_members: List[List[int]] = [[] for _ in range(num_comms)]

    for vtx in range(N):
        primary = int(comm_id[vtx].item())
        deg_primary = counts[vtx, primary].item()
        primary_members[primary].append(vtx)

        if deg_primary == 0:
            continue

        for c in range(num_comms):
            if c == primary:
                continue
            deg_c = counts[vtx, c].item()
            if deg_c < min_edges:
                continue
            if deg_c >= tau * deg_primary:
                secondary_members[c].append(vtx)

    raw_subgraphs: List[Dict[str, Any]] = []

    for k in range(num_comms):
        prim = torch.tensor(primary_members[k], dtype=torch.long, device=device)
        sec = torch.tensor(secondary_members[k], dtype=torch.long, device=device)

        if prim.numel() == 0 and sec.numel() == 0:
            continue

        core_nodes = prim
        if sec.numel() > 0:
            core_nodes = torch.unique(torch.cat([prim, sec], dim=0))

        node_ids, depths = khop_expand_nodes(
            data, core_nodes, k=khop, return_depth=True
        )
        edge_ids = induced_edges_from_nodes(data, node_ids)
        if node_ids.numel() == 0 or edge_ids.numel() == 0:
            continue

        sg: Dict[str, Any] = {
            "nodes": node_ids,
            "edges": edge_ids,
            "node_markings": None,
            "edge_markings": None,
        }

        # compute node markings
        primary_mask_global = torch.zeros(N, dtype=torch.bool, device=device)
        secondary_mask_global = torch.zeros(N, dtype=torch.bool, device=device)
        primary_mask_global[prim] = True
        secondary_mask_global[sec] = True

        is_primary = primary_mask_global[node_ids]
        is_secondary = secondary_mask_global[node_ids]
        is_core = is_primary | is_secondary
        is_halo = ~is_core

        node_markings = torch.stack(
            [is_primary.float(), is_secondary.float(), is_halo.float()],
            dim=-1,
        )
        sg["node_markings"] = node_markings

        sg["node_marking_dict"] = {
            "is_primary": is_primary.float(),
            "is_secondary": is_secondary.float(),
            "is_core": is_core.float(),
            "is_halo": is_halo.float(),
            "depth_core": depths,
        }
        sg["is_core"] = is_core
        sg["is_root"] = None
        sg["score"] = is_primary.float()


        if use_edge_markings:
            eu = u[edge_ids]
            ev = v[edge_ids]

            primary_mask_global = torch.zeros(N, dtype=torch.bool, device=device)
            secondary_mask_global = torch.zeros(N, dtype=torch.bool, device=device)
            primary_mask_global[prim] = True
            secondary_mask_global[sec] = True
            core_mask_global = primary_mask_global | secondary_mask_global

            pu = primary_mask_global[eu]
            pv = primary_mask_global[ev]
            su = secondary_mask_global[eu]
            sv = secondary_mask_global[ev]
            cu = core_mask_global[eu]
            cv = core_mask_global[ev]

            both_halo = (~cu) & (~cv)
            both_primary = pu & pv
            any_secondary = su | sv

            edge_markings = torch.stack(
                [both_primary.float(), any_secondary.float(), both_halo.float()],
                dim=-1,
            )
            sg["edge_markings"] = edge_markings

        raw_subgraphs.append(sg)

    return raw_subgraphs



# ----------------------------
# (a) Random size-matched groups policy
def clique_khop_policy(
    data: Data,
    *,
    # --- clique extraction ---
    clique_mode: Literal["all_maximal", "one_pass", "iterative"] = "iterative",
    clique_budget: Optional[int] = None,
    min_clique_size: int = 2,
    make_full_coverage: bool = True,

    # --- expansion / padding ---
    expand_mode: Literal["none", "khop", "edge_cover"] = "none",
    khop: int = 1,
    undirected_edge_cover: bool = True,

    # --- markings ---
    use_node_markings: bool = True,
    use_edge_markings: bool = True,
) -> List[Dict[str, Any]]:
    """Build overlapping subgraphs from clique cores and optional expansion.

    Args:
        data: Input graph.
        clique_mode: Strategy used to extract clique cores.
        clique_budget: Optional limit on the number of maximal cliques explored.
        min_clique_size: Minimum clique size kept as a core.
        make_full_coverage: Whether uncovered nodes should still receive a
            singleton or padded subgraph.
        expand_mode: Expansion rule applied to each clique core.
        khop: Expansion radius when ``expand_mode="khop"``.
        undirected_edge_cover: Whether edge-cover padding treats edges as
            undirected.
        use_node_markings: Whether to attach node-level subgraph markings.
        use_edge_markings: Whether to attach edge-level subgraph markings.

    Returns:
        A list of raw subgraph dictionaries describing clique-based subgraphs.
    """
    device = data.edge_index.device
    N = data.num_nodes

    # 1) Extract core clique clusters
    clusters_core = extract_clique_core_clusters(
        data,
        clique_budget=clique_budget,
        min_clique_size=min_clique_size,
        make_full_coverage=make_full_coverage,
        mode=clique_mode,
        verbose=False,
    )

    if len(clusters_core) == 0:
        # If make_full_coverage=False and no cliques found
        return []

    # 2) Expand (optional) + build per-subgraph core/halo flags + depth
    clusters_nodes: List[torch.Tensor] = []
    is_core_list: List[torch.Tensor] = []
    is_halo_list: List[torch.Tensor] = []
    depth_list: List[torch.Tensor] = []

    if expand_mode == "edge_cover":
        clusters_nodes, is_core_list, is_halo_list = edge_cover_pad_cliques(
            data,
            clusters_core,
            undirected=undirected_edge_cover,
        )
        for nodes, core_f, halo_f in zip(clusters_nodes, is_core_list, is_halo_list):
            # depth=0 for core, 1 for halo
            depth_list.append((halo_f > 0.5).long())

    elif expand_mode == "khop":
        for core_nodes in clusters_core:
            core_nodes = core_nodes.to(device)
            # khop expansion + BFS depth
            nodes, depth = khop_expand_nodes(data, core_nodes, k=khop, return_depth=True)
            if nodes.numel() == 0:
                continue

            # core/halo flags aligned with expanded `nodes`
            core_mask_global = torch.zeros(N, dtype=torch.bool, device=device)
            core_mask_global[core_nodes] = True
            is_core = core_mask_global[nodes].float()
            is_halo = (1.0 - is_core)

            clusters_nodes.append(nodes)
            is_core_list.append(is_core)
            is_halo_list.append(is_halo)
            depth_list.append(depth if depth is not None else torch.zeros_like(nodes))

    elif expand_mode == "none":
        for core_nodes in clusters_core:
            core_nodes = core_nodes.to(device)
            if core_nodes.numel() == 0:
                continue
            clusters_nodes.append(core_nodes)
            is_core = torch.ones(core_nodes.numel(), device=device)
            is_halo = torch.zeros(core_nodes.numel(), device=device)
            is_core_list.append(is_core)
            is_halo_list.append(is_halo)
            depth_list.append(torch.zeros(core_nodes.numel(), dtype=torch.long, device=device))

    else:
        raise ValueError(f"Unknown expand_mode={expand_mode}. Use 'none','khop','edge_cover'.")

    # 3) Build raw_subgraphs in your expected format
    raw_subgraphs: List[Dict[str, Any]] = []
    u, v = data.edge_index

    for nodes, core_f, halo_f, depth_core in zip(clusters_nodes, is_core_list, is_halo_list, depth_list):
        if nodes.numel() == 0:
            continue

        edge_ids = induced_edges_from_nodes(data, nodes)
        # if edge_ids.numel() == 0:
        #     continue

        sg: Dict[str, Any] = {
            "nodes": nodes,
            "edges": edge_ids,
            "node_markings": None,
            "edge_markings": None,
        }

        if use_node_markings:
            sg["node_marking_dict"] = {
                "is_core": core_f,
                "is_halo": halo_f,
                "depth_core": depth_core,
            }
            sg["is_core"] = core_f
            sg["is_root"] = None

        if use_edge_markings:
            eu = u[edge_ids]
            ev = v[edge_ids]

            core_mask_global = torch.zeros(N, dtype=torch.bool, device=device)
            core_nodes = nodes[core_f > 0.5]
            core_mask_global[core_nodes] = True

            cu = core_mask_global[eu]
            cv = core_mask_global[ev]
            core_edge = (cu & cv).float()
            cross_edge = (cu ^ cv).float()
            sg["edge_markings"] = torch.stack([core_edge, cross_edge], dim=-1)

        raw_subgraphs.append(sg)

    return raw_subgraphs


def curv_khop_policy(
    data: Data,
    khop: int = 1,
    quantile_cutoff: float = 0.8,
    threshold: Optional[float] = None,
    merge: bool = True,
    use_node_markings: bool = True,
    use_edge_markings: bool = True,
) -> List[Dict[str, Any]]:
    """Build overlapping subgraphs from curvature-based cores plus k-hop expansion.

    Args:
        data: Input graph.
        khop: Expansion radius around each curvature-derived core.
        quantile_cutoff: Quantile used when deriving the default curvature
            threshold.
        threshold: Optional explicit curvature threshold.
        merge: Whether overlapping curvature groups should be merged.
        use_node_markings: Whether to attach node-level subgraph markings.
        use_edge_markings: Whether to attach edge-level subgraph markings.

    Returns:
        A list of raw subgraph dictionaries describing curvature-based
        subgraphs.
    """
    clusters_core, debug = extract_curv_core_clusters(
        data,
        threshold=threshold,
        merge=merge,
        make_full_coverage=True,
        quantile_cutoff=quantile_cutoff
    )
    return clusters_to_raw_subgraphs(
        data,
        clusters_core,
        khop=khop,
        use_node_markings=use_node_markings,
        use_edge_markings=use_edge_markings,
        curv_debug=debug,
    )

def extract_clique_core_clusters(
    data: Data,
    clique_budget: Optional[int] = None,
    min_clique_size: int = 2,
    make_full_coverage: bool = True,
    mode: Literal["one_pass", "iterative", "all_maximal"] = "all_maximal",
    verbose: bool = False,
) -> List[torch.Tensor]:
    """
    Maximal-clique-based core clusters.

    mode:
      - "one_pass": keep each node in all max-size cliques (based on initial clique sizes)
      - "iterative": reference behavior: repeatedly remove node from non-max cliques until stable

    Notes:
      - Ties are preserved: if a node is in multiple max-size cliques, it stays in all of them.
      - Uses maximal cliques only (nx.find_cliques).
    """
    device = data.edge_index.device
    N = data.num_nodes

    # Build undirected simple graph
    ei = to_undirected(data.edge_index, num_nodes=N)
    ei, _ = remove_self_loops(ei)
    ei, _ = coalesce(ei, None, num_nodes=N)

    g_nx = nx.Graph()
    g_nx.add_nodes_from(range(N))
    g_nx.add_edges_from(ei.t().cpu().tolist())

    # Enumerate maximal cliques (budgeted)
    cliques_list: List[List[int]] = []
    for c in nx.find_cliques(g_nx):
        if len(c) < min_clique_size:
            continue
        cliques_list.append(list(c))
        if clique_budget is not None and len(cliques_list) >= clique_budget:
            if verbose:
                print(f"[clique] hit clique_budget={clique_budget}, stopping enumeration.")
            break

    # If no cliques found
    if len(cliques_list) == 0:
        if make_full_coverage:
            if verbose:
                print("[clique] no cliques found, returning singletons.")
            return [torch.tensor([i], dtype=torch.long, device=device) for i in range(N)]
        return []

    # We'll prune cliques; use mutable sets (same membership semantics)
    cliques: List[set[int]] = [set(c) for c in cliques_list]

    if mode == "one_pass":
        # ---- one-pass: compute max clique size per node from initial cliques ----
        clique_sizes = [len(c) for c in cliques]
        node_to_cliques: List[List[int]] = [[] for _ in range(N)]
        for ci, cset in enumerate(cliques):
            for n in cset:
                node_to_cliques[n].append(ci)

        max_size = [0] * N
        for n in range(N):
            if node_to_cliques[n]:
                max_size[n] = max(clique_sizes[ci] for ci in node_to_cliques[n])

        pruned: List[set[int]] = [set() for _ in range(len(cliques))]
        for n in range(N):
            ms = max_size[n]
            if ms <= 0:
                continue
            for ci in node_to_cliques[n]:
                if clique_sizes[ci] == ms:
                    pruned[ci].add(n)

        cliques = pruned

    elif mode == "iterative":
        # ---- iterative fixed-point pruning (reference while-changed logic) ----
        changed = True
        while changed:
            changed = False

            # compute node_to_cliques and max_size from CURRENT cliques
            node_to_cliques: List[List[int]] = [[] for _ in range(N)]
            max_size = [0] * N
            for ci, cset in enumerate(cliques):
                if not cset:
                    continue
                sz = len(cset)
                for n in cset:
                    node_to_cliques[n].append(ci)
                    if sz > max_size[n]:
                        max_size[n] = sz

            # remove node from any clique smaller than its current max
            for n in range(N):
                ms = max_size[n]
                if ms <= 0:
                    continue
                for ci in node_to_cliques[n]:
                    if len(cliques[ci]) < ms:
                        if n in cliques[ci]:
                            cliques[ci].remove(n)
                            changed = True
    elif mode == "all_maximal":
        pass # keep all maximal cliques as-is
    else:
        raise ValueError(f"Unknown mode={mode}. Use 'one_pass' or 'iterative'.")

    # Remove empty cliques and duplicates
    unique: List[frozenset[int]] = []
    for cset in cliques:
        if len(cset) == 0:
            continue
        fs = frozenset(cset)
        if fs not in unique:
            unique.append(fs)

    clusters = [torch.tensor(sorted(list(fs)), dtype=torch.long, device=device) for fs in unique]

    # Full coverage: add singleton clusters for uncovered nodes
    if make_full_coverage:
        covered = torch.zeros(N, dtype=torch.bool, device=device)
        for c in clusters:
            covered[c] = True
        for i in range(N):
            if not covered[i]:
                if verbose:
                    print(f"[clique] adding singleton for uncovered node {i}.")
                clusters.append(torch.tensor([i], dtype=torch.long, device=device))

    return clusters


def edge_cover_pad_cliques(
    data: Data,
    clusters_core: List[torch.Tensor],
    *,
    undirected: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Expand clique clusters minimally to cover *all* original edges.

    For each missing edge (u,v) that is not contained in any cluster:
      - add v to all clusters containing u
      - add u to all clusters containing v
    Newly added nodes are marked as halo for those clusters.

    Returns:
      clusters_expanded: List[LongTensor] nodes in expanded cluster (core ∪ halo)
      is_core_list:      List[FloatTensor] [|cluster|] 1 for core nodes else 0
      is_halo_list:      List[FloatTensor] [|cluster|] 1 for halo nodes else 0
    """
    device = data.edge_index.device
    N = data.num_nodes

    # Make edges undirected unique if requested (recommended for "edge coverage")
    edge_index = data.edge_index
    if undirected:
        ei = to_undirected(edge_index, num_nodes=N)
        ei, _ = remove_self_loops(ei)
        ei, _ = coalesce(ei, None, num_nodes=N)
    else:
        ei = edge_index

    u_all, v_all = ei[0], ei[1]
    E = u_all.numel()

    # Represent clusters as python sets for fast membership / insertion
    core_sets = [set(c.tolist()) for c in clusters_core]
    exp_sets  = [set(c.tolist()) for c in clusters_core]  # mutable expansions

    # Node -> list of clusters where node is currently in CORE (not halo)
    # (Important: use CORE membership here, per your description of "incident cliques")
    node_to_core_cliques = [[] for _ in range(N)]
    for ci, c in enumerate(core_sets):
        for n in c:
            node_to_core_cliques[n].append(ci)

    # Quick check for whether any cluster already contains both endpoints.
    # We'll do it via intersecting the per-node core-clique lists first (cheap),
    # and only fall back to set membership if needed.
    def edge_is_covered(u: int, v: int) -> bool:
        cu = node_to_core_cliques[u]
        cv = node_to_core_cliques[v]
        if len(cu) == 0 or len(cv) == 0:
            return False
        # If they share a clique id, edge is definitely covered in the core
        # (and thus in expanded too).
        # Use a small-set intersection:
        if len(cu) < len(cv):
            s = set(cu)
            return any(x in s for x in cv)
        else:
            s = set(cv)
            return any(x in s for x in cu)

    # Track whether we actually added anything
    num_repairs = 0

    # Iterate edges; if uncovered, expand incident cliques
    for uu, vv in zip(u_all.tolist(), v_all.tolist()):
        if uu == vv:
            continue

        if edge_is_covered(uu, vv):
            continue

        # uncovered -> add vv to all CORE cliques of uu
        for ci in node_to_core_cliques[uu]:
            if vv not in exp_sets[ci]:
                exp_sets[ci].add(vv)
                num_repairs += 1

        # uncovered -> add uu to all CORE cliques of vv
        for ci in node_to_core_cliques[vv]:
            if uu not in exp_sets[ci]:
                exp_sets[ci].add(uu)
                num_repairs += 1

    # Build output tensors + core/halo flags aligned with each cluster's node list
    clusters_expanded: List[torch.Tensor] = []
    is_core_list: List[torch.Tensor] = []
    is_halo_list: List[torch.Tensor] = []

    for ci in range(len(exp_sets)):
        nodes = sorted(list(exp_sets[ci]))
        nodes_t = torch.tensor(nodes, dtype=torch.long, device=device)

        core_mask = torch.tensor([1.0 if n in core_sets[ci] else 0.0 for n in nodes], device=device)
        halo_mask = 1.0 - core_mask

        clusters_expanded.append(nodes_t)
        is_core_list.append(core_mask)
        is_halo_list.append(halo_mask)

    return clusters_expanded, is_core_list, is_halo_list



# ============================================================
# Curvature (Balanced Forman Curvature) extractor (core-only)
# ============================================================

def _most_frequent_count(items: List[int]) -> int:
    if not items:
        return 0
    # count of the mode
    m = max(set(items), key=items.count)
    return items.count(m)


def balanced_forman_curvature(edges_undirected: List[Tuple[int, int]]) -> List[float]:
    """
    Directly adapted from EXPRESSIVE POOLING by Lachi et al. (with undirected unique edges expected: u < v).
    Returns one curvature per edge in edges_undirected (same order).
    """
    if edges_undirected is None or len(edges_undirected) == 0:
        return []

    num_nodes = max(max(u, v) for (u, v) in edges_undirected) + 1

    candidates = [[] for _ in range(num_nodes)]
    for u, v in edges_undirected:
        candidates[u].append(v)
        candidates[v].append(u)

    degrees = [len(candidates[i]) for i in range(num_nodes)]

    triangles = []
    quads_i, quads_j = [], []
    quads_i_set, quads_j_set = [], []

    for (i, j) in edges_undirected:
        tri = list(set(candidates[i]).intersection(set(candidates[j])))
        triangles.append(tri)

        qi_temp = []
        qi_temp_set = set()
        # nodes connected to i but not to j
        for k in candidates[i]:
            if k not in candidates[j] and k != i and k != j:
                # omega neighbors shared by k and j, but omega not connected to i
                for omega in set(candidates[k]).intersection(candidates[j]):
                    if omega not in candidates[i] and omega != i and omega != j:
                        qi_temp.append((k, omega))
                        qi_temp_set.add(k)
        quads_i.append(qi_temp)
        quads_i_set.append(frozenset(qi_temp_set))

        qj_temp = []
        qj_temp_set = set()
        for k in candidates[j]:
            if k not in candidates[i] and k != i and k != j:
                for omega in set(candidates[k]).intersection(candidates[i]):
                    if omega not in candidates[j] and omega != i and omega != j:
                        qj_temp.append((k, omega))
                        qj_temp_set.add(k)
        quads_j.append(qj_temp)
        quads_j_set.append(frozenset(qj_temp_set))

    num_quads = []
    for t in range(len(edges_undirected)):
        counts_i = [x for quad in quads_i[t] for x in quad]
        counts_j = [x for quad in quads_j[t] for x in quad]
        num_quads.append(max(_most_frequent_count(counts_i), _most_frequent_count(counts_j)))

    bfc = []
    for idx, (i, j) in enumerate(edges_undirected):
        min_degree = min(degrees[i], degrees[j])
        max_degree = max(degrees[i], degrees[j])

        if min_degree == 1:
            result = 0.0
        elif len(quads_i[idx]) == 0:
            result = (
                2 / degrees[i]
                + 2 / degrees[j]
                - 2
                + 2 * len(triangles[idx]) / max_degree
                + len(triangles[idx]) / min_degree
            )
        else:
            result = (
                2 / degrees[i]
                + 2 / degrees[j]
                - 2
                + 2 * len(triangles[idx]) / max_degree
                + len(triangles[idx]) / min_degree
                + (num_quads[idx] ** (-1)) / max_degree * (len(quads_i_set[idx]) + len(quads_j_set[idx]))
            )
        bfc.append(float(result))

    return bfc


def _merge_common_sets(groups: List[set[int]]) -> List[List[int]]:
    """
    Merge all groups that share at least one element (connected components in overlap graph).
    Output is list of sorted node lists.
    """
    if not groups:
        return []

    neigh = defaultdict(set)
    for each in groups:
        for item in each:
            neigh[item].update(each)

    visited = set()

    def component(start: int):
        stack = [start]
        comp = set()
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            comp.add(n)
            for nb in neigh[n]:
                if nb not in visited:
                    stack.append(nb)
        return comp

    merged = []
    for node in neigh:
        if node not in visited:
            merged.append(sorted(list(component(node))))
    return merged


def extract_curv_core_clusters(
    data: Data,
    threshold: Optional[float] = None,
    merge: bool = True,
    make_full_coverage: bool = True,
    return_debug: bool = True,
    quantile_cutoff: float = 0.8
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], Dict[str, Any]]]:
    """
    Curv core cluster extractor with optional debug payload.

    Returns clusters as List[LongTensor].
    If return_debug=True, also returns a debug dict containing BOTH:
      - mapping/set form (fast lookup):
          * bfc_by_undirected_edge: dict[(u,v) with u<v -> float]
          * selected_edges_set: set[(u,v) with u<v]
      - array form (easy to inspect / plot):
          * selected_edges_uv: LongTensor [2, M] with u<v
          * selected_bfc: FloatTensor [M]
      - plus:
          * threshold: float (the value used)
          * all_edges_uv: LongTensor [2, E_undirected]
          * all_bfc: FloatTensor [E_undirected]
    """
    device = data.edge_index.device
    N = data.num_nodes

    # ---- build undirected unique edge list (u < v) ----
    ei = to_undirected(data.edge_index, num_nodes=N)
    ei, _ = remove_self_loops(ei)
    ei, _ = coalesce(ei, None, num_nodes=N)

    u, v = ei[0].cpu().tolist(), ei[1].cpu().tolist()
    edges_set = set()
    for a, b in zip(u, v):
        if a == b:
            continue
        aa, bb = (a, b) if a < b else (b, a)
        edges_set.add((aa, bb))
    edges = sorted(list(edges_set))  # List[(u,v)] with u<v

    # ---- curvature per undirected edge ----
    bfc = balanced_forman_curvature(edges)  # List[float], aligned with edges

    if threshold is None:
        if quantile_cutoff == "mean":
            threshold_val = (sum(bfc) / len(bfc)) if len(bfc) > 0 else 0.0
        else:
            # use quantile
            threshold_val = (
                float(torch.quantile(torch.tensor(bfc, dtype=torch.float), quantile_cutoff))
                if len(bfc) > 0
                else 0.0
            )
    else:
        threshold_val = float(threshold)

    # ---- debug mapping/set forms ----
    bfc_by_undirected_edge = {edges[i]: float(bfc[i]) for i in range(len(edges))}
    selected_idx = [i for i in range(len(edges)) if float(bfc[i]) > threshold_val]
    selected_edges_set = {edges[i] for i in selected_idx}

    # ---- groups from selected edges ----
    groups = [set([edges[i][0], edges[i][1]]) for i in selected_idx]

    if merge:
        merged_lists = _merge_common_sets(groups)  # List[List[int]]
        clusters = [torch.tensor(lst, dtype=torch.long, device=device) for lst in merged_lists]
    else:
        clusters = [torch.tensor(sorted(list(s)), dtype=torch.long, device=device) for s in groups]

    # ---- full coverage (add singleton for every uncovered node) ----
    if make_full_coverage:
        covered = torch.zeros(N, dtype=torch.bool, device=device)
        for c in clusters:
            if c.numel() > 0:
                covered[c] = True
        for i in range(N):
            if not covered[i]:
                clusters.append(torch.tensor([i], dtype=torch.long, device=device))

    if not return_debug:
        return clusters

    # ---- debug array forms ----
    if len(edges) > 0:
        all_edges_uv = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()  # [2, E]
        all_bfc = torch.tensor([float(x) for x in bfc], dtype=torch.float, device=device)     # [E]
    else:
        all_edges_uv = torch.empty((2, 0), dtype=torch.long, device=device)
        all_bfc = torch.empty((0,), dtype=torch.float, device=device)

    if len(selected_idx) > 0:
        sel_edges = [edges[i] for i in selected_idx]
        sel_bfc = [float(bfc[i]) for i in selected_idx]
        selected_edges_uv = torch.tensor(sel_edges, dtype=torch.long, device=device).t().contiguous()  # [2, M]
        selected_bfc = torch.tensor(sel_bfc, dtype=torch.float, device=device)                          # [M]
    else:
        selected_edges_uv = torch.empty((2, 0), dtype=torch.long, device=device)
        selected_bfc = torch.empty((0,), dtype=torch.float, device=device)

    debug = {
        "threshold": threshold_val,
        # fast lookup
        "bfc_by_undirected_edge": bfc_by_undirected_edge,  # dict[(u,v)->float], u<v
        "selected_edges_set": selected_edges_set,          # set[(u,v)], u<v
        # convenient arrays
        "all_edges_uv": all_edges_uv,                      # [2, E_undirected], u<v
        "all_bfc": all_bfc,                                # [E_undirected]
        "selected_edges_uv": selected_edges_uv,            # [2, M], u<v
        "selected_bfc": selected_bfc,                      # [M]
    }
    return clusters, debug


# ============================================================
# Build raw_subgraphs from core clusters with padding + markings for pooling schemes
# ============================================================

def clusters_to_raw_subgraphs(
    data: Data,
    clusters_core: List[torch.Tensor],
    khop: int = 1,
    use_node_markings: bool = True,
    use_edge_markings: bool = True,
    curv_debug: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Convert node clusters into the raw-subgraph format used downstream.

    Args:
        data: Input graph.
        clusters_core: List of node sets defining the subgraph cores.
        khop: Expansion radius around each core before inducing edges.
        use_node_markings: Whether to attach node-level subgraph markings.
        use_edge_markings: Whether to attach edge-level subgraph markings.
        curv_debug: Optional curvature metadata to align with induced edges.

    Returns:
        A list of raw subgraph dictionaries consumed by ``SubgraphTransform``.
    """
    device = data.edge_index.device
    N = data.num_nodes
    ei = data.edge_index

    raw: List[Dict[str, Any]] = []

    for core_nodes in clusters_core:
        core_nodes = core_nodes.to(device).unique()
        if core_nodes.numel() == 0:
            continue

        node_ids, depths = khop_expand_nodes(data, core_nodes, k=khop, return_depth=True)
        edge_ids = induced_edges_from_nodes(data, node_ids)

        # ---- Curvature debug (optional): per induced edge ----
        # Aligned with edge_ids (indices into data.edge_index)
        if curv_debug is not None:
            bfc_map = curv_debug["bfc_by_undirected_edge"]  # dict[(u,v)->float], u<v
            sel_set = curv_debug["selected_edges_set"]      # set[(u,v)], u<v

            u_all, v_all = data.edge_index
            eu = u_all[edge_ids]
            ev = v_all[edge_ids]

            a = torch.minimum(eu, ev).tolist()
            b = torch.maximum(eu, ev).tolist()

            bfc_vals = torch.tensor(
                [bfc_map.get((ai, bi), 0.0) for ai, bi in zip(a, b)],
                dtype=torch.float,
                device=device,
            )
            is_selected = torch.tensor(
                [1.0 if (ai, bi) in sel_set else 0.0 for ai, bi in zip(a, b)],
                dtype=torch.float,
                device=device,
            )

            # [E_sub, 2] = [bfc_value, selected_flag]
            edge_debug = torch.stack([bfc_vals, is_selected], dim=-1)
        else:
            edge_debug = None

        if node_ids.numel() == 0:
            continue
        # Keep degenerate node-only subgraphs; downstream packaging handles them.
        sg: Dict[str, Any] = {
            "nodes": node_ids,
            "edges": edge_ids,
            "node_markings": None,
            "edge_markings": None,
            "edge_debug": edge_debug,
        }

        if use_node_markings:
            core_mask_global = torch.zeros(N, dtype=torch.bool, device=device)
            core_mask_global[core_nodes] = True

            is_core = core_mask_global[node_ids].float()
            is_halo = (~core_mask_global[node_ids]).float()

            sg["node_markings"] = torch.stack([is_core, is_halo], dim=-1)

            sg["node_marking_dict"] = {
                "is_core": is_core,
                "is_halo": is_halo,
                "depth_core": depths,  # 0 for core, 1..khop for halo
            }
            sg["is_core"] = is_core
            sg["is_root"] = None
            sg["score"] = None

        if use_edge_markings:
            # Edge markings based on core membership (core/core vs boundary)
            u, v = data.edge_index
            eu = u[edge_ids]
            ev = v[edge_ids]

            core_mask_global = torch.zeros(N, dtype=torch.bool, device=device)
            core_mask_global[core_nodes] = True
            cu = core_mask_global[eu]
            cv = core_mask_global[ev]

            core_edge = (cu & cv).float()
            cross_edge = (cu ^ cv).float()
            sg["edge_markings"] = torch.stack([core_edge, cross_edge], dim=-1)

        raw.append(sg)

    return raw

