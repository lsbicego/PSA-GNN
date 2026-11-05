import torch
from torch import Tensor
from torch_geometric.data import Data
from typing import List, Dict, Optional, Tuple, Any
from collections import deque

from .subgraph_utils import bfs_dist

# ======================================================================
# === Utility: adjacency list construction (CPU) ========================
# ======================================================================

def depth_root_bfs(edge_index_cpu, nodes, edges, s_int, device_graph, max_k=5):
    # NOTE: max_k is set to 5
    # recover subgraph edges to do bfs_dist
    subgraph_edges = edge_index_cpu[:, edges.cpu()]

    # make subgraph_edges in local indexing, i.e., start at 0
    node_id_map = {int(n.item()): i for i, n in enumerate(nodes.cpu())}
    u_sg = subgraph_edges[0].apply_(lambda x: node_id_map[int(x)])
    v_sg = subgraph_edges[1].apply_(lambda x: node_id_map[int(x)])
    subgraph_edges = torch.stack([u_sg, v_sg], dim=0).to(device_graph)

    # traanslate root to local indexing
    local_root = node_id_map[s_int]

    depth_root = bfs_dist(subgraph_edges, nodes.cpu(), local_root, max_k=max_k).to(device_graph)
    return depth_root


def build_adj_and_deg(data: Data):
    """
    Return:
      adj: List[List[int]]
      deg: Tensor float32, deg>=1
      edge_index_cpu: LongTensor on CPU
    """
    edge_index_cpu = data.edge_index.cpu()
    u = edge_index_cpu[0]
    v = edge_index_cpu[1]
    N = data.num_nodes

    adj = [[] for _ in range(N)]
    E = edge_index_cpu.size(1)
    for e in range(E):
        a = int(u[e])
        b = int(v[e])
        adj[a].append(b)
        adj[b].append(a)

    deg = torch.tensor([len(nbrs) for nbrs in adj], dtype=torch.float32)
    deg = deg.clamp(min=1.0)
    return adj, deg, edge_index_cpu


# ======================================================================
# === Push-Based ACL PPR ===============================================
# ======================================================================

def local_push_ppr(adj: List[List[int]],
                   deg: Tensor,
                   seed: int,
                   alpha: float = 0.15,
                   eps: float = 1e-4) -> Tuple[Tensor, Tensor]:

    N = deg.size(0)
    p = torch.zeros(N, dtype=torch.float32)
    r = torch.zeros(N, dtype=torch.float32)
    r[seed] = 1.0

    Q = deque()
    in_queue = torch.zeros(N, dtype=torch.bool)

    if r[seed] / deg[seed] > eps:
        Q.append(seed)
        in_queue[seed] = True

    while Q:
        vtx = Q.popleft()
        in_queue[vtx] = False

        rv = r[vtx]
        if rv <= 0 or rv / deg[vtx] <= eps:
            continue

        # push
        p[vtx] += alpha * rv
        res = (1 - alpha) * rv
        r[vtx] = res / 2
        inc = res / (2 * deg[vtx]) if deg[vtx] > 0 else 0.0

        for nbr in adj[vtx]:
            r[nbr] += inc
            if r[nbr] / deg[nbr] > eps and not in_queue[nbr]:
                Q.append(nbr)
                in_queue[nbr] = True

    return p, r


# ======================================================================
# === Power Iteration PPR (CPU or CUDA) ================================
# ======================================================================
def local_power_ppr(edge_index: Tensor,
                    deg: Tensor,
                    seed: int,
                    alpha: float = 0.15,
                    num_iters: int = 20,
                    tol: Optional[float] = None) -> Tuple[Tensor, Tensor]:
    """
    Local personalized PageRank via power iteration.

    Assumes:
        - edge_index is on the desired device (CPU or CUDA)
        - deg is on the same device as edge_index
    """
    u = edge_index[0]
    v = edge_index[1]
    device = edge_index.device
    N = deg.size(0)

    e = torch.zeros(N, dtype=torch.float32, device=device)
    e[seed] = 1.0
    p = e.clone()
    r = torch.zeros_like(p)

    for _ in range(num_iters):
        q = torch.zeros_like(p)
        q.index_add_(0, v, p[u] / deg[u])

        p_next = alpha * e + (1 - alpha) * q
        r = p_next - p

        if tol is not None:
            if torch.norm(r, p=1) <= tol:
                p = p_next
                break

        p = p_next

    return p, r



# ======================================================================
# === Hybrid Sweep Cut (Option C) =======================================
# ======================================================================

def hybrid_sweep_cut(adj: List[List[int]],
                     deg: Tensor,
                     p: Tensor,
                     vol_min_frac: float = 0.05,
                     vol_max_frac: float = 0.20,
                     penalty_lambda: float = 0.0) -> Tensor:
    """
    Hybrid strategy:
      1. Do full sweep (compute conductance for all prefixes)
      2. Record the true minimum conductance prefix S_best
      3. If its volume is within [vol_min, vol_max], accept it
      4. Else:
          - among prefixes satisfying volume constraints,
            pick the one with minimum conductance
          - if none exist, fallback:
              pick prefix whose volume is closest to mid-volume
    """

    N = deg.size(0)
    total_vol = float(deg.sum().item())
    vol_min = vol_min_frac * total_vol
    vol_max = vol_max_frac * total_vol

    active = (p > 0).nonzero(as_tuple=False).view(-1)
    if active.numel() == 0:
        return active

    scores = p[active] / deg[active]
    scores, idx = torch.sort(scores, descending=True)
    order = active[idx]

    in_set = torch.zeros(N, dtype=torch.bool)
    vol_S = 0.0
    cut_S = 0.0

    best_phi = float("inf")
    best_k = 0

    phi_list = []
    vol_list = []

    for k, vtx in enumerate(order, start=1):
        vi = int(vtx)

        # add node
        in_set[vi] = True
        vol_S += deg[vi].item()

        # update cut
        for nbr in adj[vi]:
            if in_set[nbr]:
                cut_S -= 1
            else:
                cut_S += 1

        # compute conductance
        if 0 < vol_S < total_vol:
            phi = cut_S / min(vol_S, total_vol - vol_S)
        else:
            phi = float("inf")

        vol_list.append(vol_S)
        phi_list.append(phi)

        if phi < best_phi:
            best_phi = phi
            best_k = k

    # Step 1: true best
    S_true = order[:best_k]
    vol_true = vol_list[best_k - 1]

    if vol_min <= vol_true <= vol_max:
        return S_true

    # Step 2: best feasible prefix
    feasible_idxs = [i for i, vol_i in enumerate(vol_list)
                     if vol_min <= vol_i <= vol_max]

    if feasible_idxs:
        best_i = min(feasible_idxs, key=lambda i: phi_list[i])
        return order[: best_i + 1]

    # Step 3: fallback: closest to mid target
    target_vol = 0.5 * (vol_min + vol_max)
    best_i = min(range(len(vol_list)),
                 key=lambda i: abs(vol_list[i] - target_vol))
    return order[: best_i + 1]


# ======================================================================
# === Seeds: degree-based, random, farthest, coarse cluster ============
# ======================================================================

def select_degree_seeds(data: Data, num_seeds: int) -> Tensor:
    deg = torch.bincount(
        torch.cat([data.edge_index[0], data.edge_index[1]]),
        minlength=data.num_nodes,
    ).float()
    idx = torch.argsort(deg, descending=True)
    return idx[: min(num_seeds, data.num_nodes)]


def select_random_seeds(data: Data, num_seeds: int) -> Tensor:
    num = min(num_seeds, data.num_nodes)
    return torch.randperm(data.num_nodes)[:num]


def select_farthest_point_seeds(data: Data, num_seeds: int) -> Tensor:
    """
    Graph-distance farthest-point seeding:
      start from max-degree node,
      then repeatedly pick node maximizing distance to closest seed.
    """
    N = data.num_nodes
    u = data.edge_index[0]
    v = data.edge_index[1]

    adj = [[] for _ in range(N)]
    for a, b in zip(u.tolist(), v.tolist()):
        adj[a].append(b)
        adj[b].append(a)

    deg = torch.bincount(torch.cat([u, v]), minlength=N).float()
    seeds = [int(torch.argmax(deg))]

    def bfs_dist(s):
        dist = [-1] * N
        dist[s] = 0
        Q = deque([s])
        while Q:
            x = Q.popleft()
            for nbr in adj[x]:
                if dist[nbr] < 0:
                    dist[nbr] = dist[x] + 1
                    Q.append(nbr)
        return dist

    dists = [bfs_dist(seeds[0])]

    while len(seeds) < num_seeds:
        min_dist = [min(d[i] for d in dists if d[i] >= 0) for i in range(N)]

        # ignore seeds
        for s in seeds:
            min_dist[s] = -1

        nxt = max(range(N), key=lambda i: min_dist[i])
        if min_dist[nxt] <= 0:
            break

        seeds.append(nxt)
        dists.append(bfs_dist(nxt))

    return torch.tensor(seeds, dtype=torch.long)


# ======================================================================
# === Subgraph Extraction Helpers ======================================
# ======================================================================

def bfs_khop(data: Data,
             core: Tensor,
             k: int) -> Tuple[Tensor, Tensor]:
    """
    BFS outward from 'core' nodes up to k hops.
    Returns:
      all_nodes: LongTensor of core+halo
      depths: depth for each node in all_nodes
    """
    if k == 0:
        depths = torch.zeros(core.size(0), dtype=torch.long)
        return core, depths

    N = data.num_nodes
    adj = [[] for _ in range(N)]
    u, v = data.edge_index
    for a, b in zip(u.tolist(), v.tolist()):
        adj[a].append(b)
        adj[b].append(a)

    visited = {int(n): 0 for n in core.tolist()}
    Q = deque([(int(n), 0) for n in core.tolist()])

    while Q:
        node, d = Q.popleft()
        if d == k:
            continue
        for nbr in adj[node]:
            if nbr not in visited:
                visited[nbr] = d + 1
                Q.append((nbr, d + 1))

    nodes = torch.tensor(list(visited.keys()), dtype=torch.long)
    depths = torch.tensor([visited[int(n)] for n in nodes], dtype=torch.long)
    return nodes, depths


def edges_from_nodes(data: Data, nodes: Tensor) -> Tensor:
    mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    mask[nodes] = True
    u, v = data.edge_index
    keep = mask[u] & mask[v]
    return keep.nonzero(as_tuple=False).view(-1)


# ======================================================================
# === Main ACL Policy with Hybrid Sweep + Adaptive PPR Solver ==========
# ======================================================================
def acl_policy(
    data: Data,
    seed_mode: str = "farthest",
    max_seeds: Optional[int] = None,
    alpha: float = 0.15,
    eps: float = 1e-4,
    khop: int = 1,
    vol_min_frac: float = 0.05,
    vol_max_frac: float = 0.20,
    ppr_num_iters: int = 20,
    ppr_tol: float = 1e-6,
    max_rescue: Optional[int] = None,
    use_node_markings: bool = True,
    use_edge_markings: bool = True,
    chosen_solver: str = None,
    ppr_use_cuda: bool = False,          # NEW
) -> Dict[str, List]:

    import time
    device_graph = data.edge_index.device  # usually cpu in your setup
    N = data.num_nodes
    setattr(acl_policy, "chosen_ppr_solver", chosen_solver)

    # ------------------------------------------------------------
    # 1) Build adjacency + deg on CPU (needed for push & sweep)
    # ------------------------------------------------------------
    adj, deg_cpu, edge_index_cpu = build_adj_and_deg(data)

    # ------------------------------------------------------------
    # 2) Decide PPR device for power iteration
    # ------------------------------------------------------------
    if ppr_use_cuda and torch.cuda.is_available():
        ppr_device = torch.device("cuda")
    else:
        ppr_device = torch.device("cpu")

    edge_index_ppr = edge_index_cpu.to(ppr_device)
    deg_ppr = deg_cpu.to(ppr_device)

    # ------------------------------------------------------------
    # 3) Pick initial seeds (as before)
    # ------------------------------------------------------------
    if max_seeds is None:
        max_seeds = max(1, N // 5)

    seed_mode = seed_mode.lower()
    if seed_mode == "degree":
        seeds = select_degree_seeds(data, max_seeds)
    elif seed_mode == "random":
        seeds = select_random_seeds(data, max_seeds)
    elif seed_mode == "farthest":
        seeds = select_farthest_point_seeds(data, max_seeds)
    else:
        raise ValueError(f"Unknown seed_mode={seed_mode}")

    seeds = seeds.to(device_graph)

    # ------------------------------------------------------------
    # 4) Adaptive solver choice (first call only)
    # ------------------------------------------------------------
    if not hasattr(acl_policy, "chosen_ppr_solver"):
        seed0 = int(seeds[0].item())

        # push timing (CPU only)
        t0 = time.time()
        _ = local_push_ppr(adj, deg_cpu, seed0, alpha=alpha, eps=eps)
        push_t = time.time() - t0

        # power timing (on chosen PPR device, including transfer already done)
        t0 = time.time()
        _ = local_power_ppr(edge_index_ppr, deg_ppr, seed0,
                            alpha=alpha, num_iters=ppr_num_iters, tol=ppr_tol)
        power_t = time.time() - t0

        if power_t < push_t:
            setattr(acl_policy, "chosen_ppr_solver", "power")
        else:
            setattr(acl_policy, "chosen_ppr_solver", "push")
        selected_solver = getattr(acl_policy, "chosen_ppr_solver")
        print(f"ACL iterative policy selected PPR solver: {selected_solver}")
    
    solver = getattr(acl_policy, "chosen_ppr_solver")
    

    # ------------------------------------------------------------
    # 5) Main ACL loop over seeds
    # ------------------------------------------------------------
    raw_subgraphs = []
    covered = torch.zeros(N, dtype=torch.bool)

    for s in seeds:
        s_int = int(s.item())

        # --- compute PPR
        if solver == "push":
            p_cpu, _ = local_push_ppr(adj, deg_cpu, s_int, alpha=alpha, eps=eps)
            p = p_cpu.to(device_graph)
        else:
            p, _ = local_power_ppr(edge_index_ppr, deg_ppr, s_int,
                                   alpha=alpha, num_iters=ppr_num_iters, tol=ppr_tol)
            p_cpu = p.cpu()

        # --- hybrid sweep cut
        core_cpu = hybrid_sweep_cut(adj, deg_cpu, p_cpu,
                                    vol_min_frac=vol_min_frac,
                                    vol_max_frac=vol_max_frac)
        if core_cpu.numel() == 0:
            continue
        core = core_cpu.to(device_graph)

        # --- k-hop halo
        nodes, depths = bfs_khop(data, core, k=khop)
        nodes = nodes.to(device_graph)
        depths = depths.to(device_graph)

        edges = edges_from_nodes(data, nodes).to(device_graph)
        if nodes.numel() == 0 or edges.numel() == 0:
            continue

        covered[nodes.cpu()] = True

        # markings
        is_seed = (nodes == s_int).float()
        pvals = p_cpu[nodes]
        node_mark = torch.stack([is_seed, pvals, depths.float()], dim=-1)

        # NOTE: max_k is set to 5
        depth_root = depth_root_bfs(edge_index_cpu, nodes, edges, s_int, device_graph, max_k=5)
        
        node_marking_dict = {
            "is_root": is_seed,
            "is_core": (nodes.unsqueeze(1) == core.unsqueeze(0)).any(dim=1).float(),
            "depth_root": depth_root,
            "depth_core": depths,
            "ppr_value": pvals,
            "ppr_normalized": pvals / (pvals.sum() + 1e-15),
            "ppr_degree_normalized": pvals / (deg_cpu[nodes.cpu()].clamp_min(1.0).to(device_graph)),
        }


        if use_edge_markings:
            u, v = data.edge_index
            e_u = u[edges]
            e_v = v[edges]
            p_u = p[e_u]
            p_v = p[e_v]
            e_mark = torch.stack(
                [torch.minimum(p_u, p_v), torch.maximum(p_u, p_v)], dim=-1
            )
        else:
            e_mark = None

        raw_subgraphs.append(
            {
                "nodes": nodes,
                "edges": edges,
                "node_markings": node_mark,
                "edge_markings": e_mark,
                "node_marking_dict": node_marking_dict,
                "is_root": is_seed,
                "is_core": node_marking_dict["is_core"],
                "score": node_marking_dict["ppr_value"],
            }
        )

    # ------------------------------------------------------------
    # 6) Rescue phase (unchanged, stays on CPU/graph device)
    # ------------------------------------------------------------
    uncovered = (~covered).nonzero(as_tuple=False).view(-1)
    if uncovered.numel() > 0:
        if max_rescue is None:
            max_rescue = uncovered.numel()
        rescued = 0

        for u0 in uncovered.tolist():
            if rescued >= max_rescue:
                break

            s_int = int(u0)
            core = torch.tensor([s_int], dtype=torch.long, device=device_graph)

            nodes, depths = bfs_khop(data, core, k=1)
            nodes = nodes.to(device_graph)
            depths = depths.to(device_graph)
            edges = edges_from_nodes(data, nodes).to(device_graph)

            covered[nodes.cpu()] = True

            # compute markings
            is_seed = (nodes == s_int).float()
            pvals = torch.zeros_like(is_seed)
            node_mark = torch.stack([is_seed, pvals, depths.float()], dim=-1)

            # NOTE: max_k is set to 5
            depth_root = depth_root_bfs(edge_index_cpu, nodes, edges, s_int, device_graph, max_k=5)

            node_marking_dict = {
            "is_root": is_seed,
            "is_core": (nodes.unsqueeze(1) == core.unsqueeze(0)).any(dim=1).float(),
            "depth_root": depth_root,
            "depth_core": depths,
            "ppr_value": pvals,
            "ppr_normalized": pvals,
            "ppr_degree_normalized": pvals,
            }


            if use_edge_markings:
                u, v = data.edge_index
                e_u = u[edges]
                e_v = v[edges]
                zeros = torch.zeros_like(e_u, dtype=torch.float32)
                e_mark = torch.stack([zeros, zeros], dim=-1)
            else:
                e_mark = None

            raw_subgraphs.append(
                {
                    "nodes": nodes,
                    "edges": edges,
                    "node_markings": node_mark,
                    "edge_markings": e_mark,
                    "node_marking_dict": node_marking_dict,
                    "is_root": is_seed,
                    "is_core": node_marking_dict["is_core"],
                    "score": node_marking_dict["ppr_value"],
                }
            )
            rescued += 1

    return raw_subgraphs


def acl_policy_iterative(
    data: Data,
    seed_mode: str = "degree_iter",    # "degree_iter" or "random_iter"
    max_subgraphs: Optional[int] = None,
    coverage_mode: str = "core_halo",   # "core_halo" or "full_halo"
    alpha: float = 0.15,
    eps: float = 1e-4,
    khop: int = 1,
    vol_min_frac: float = 0.05,
    vol_max_frac: float = 0.20,
    ppr_num_iters: int = 20,
    ppr_tol: float = 1e-6,
    max_rescue: Optional[int] = None,
    use_node_markings: bool = True,
    use_edge_markings: bool = True,
    ppr_use_cuda: bool = False,        # NEW
    chosen_solver: str = "",
) -> Dict[str, List]:
    """
    Iterative ACL policy (as before), with an option to run the
    power-based PPR on CUDA if available.
    """
    import time

    device_graph = data.edge_index.device
    N = data.num_nodes
    setattr(acl_policy, "chosen_ppr_solver", chosen_solver)

    # 1) CPU adjacency + degree
    adj, deg_cpu, edge_index_cpu = build_adj_and_deg(data)

    # 2) PPR device
    if ppr_use_cuda and torch.cuda.is_available():
        ppr_device = torch.device("cuda")
    else:
        ppr_device = torch.device("cpu")

    edge_index_ppr = edge_index_cpu.to(ppr_device)
    deg_ppr = deg_cpu.to(ppr_device)

    # 3) max_subgraphs
    if max_subgraphs is None:
        max_subgraphs = max(1, N // 5)

    # 4) adaptive solver (reuse from acl_policy if set)
    solver = None
    if hasattr(acl_policy, "chosen_ppr_solver"):
        solver = getattr(acl_policy, "chosen_ppr_solver")
    if not solver:
        seed0 = int(torch.argmax(deg_cpu).item())

        t0 = time.time()
        _ = local_push_ppr(adj, deg_cpu, seed0, alpha=alpha, eps=eps)
        push_t = time.time() - t0

        t0 = time.time()
        _ = local_power_ppr(edge_index_ppr, deg_ppr, seed0,
                            alpha=alpha, num_iters=ppr_num_iters, tol=ppr_tol)
        power_t = time.time() - t0

        solver = "power" if power_t < push_t else "push"
        setattr(acl_policy, "chosen_ppr_solver", solver)
        # print(f"ACL iterative policy selected PPR solver: {solver}")

    # 5) iterative seeding
    covered = torch.zeros(N, dtype=torch.bool)
    raw_subgraphs: List[Dict] = []

    def pick_next_seed() -> Optional[int]:
        uncovered = (~covered).nonzero(as_tuple=False).view(-1)
        if uncovered.numel() == 0:
            return None

        if seed_mode == "degree_iter":
            cand_deg = deg_cpu[uncovered]
            return int(uncovered[torch.argmax(cand_deg)].item())
        elif seed_mode == "random_iter":
            idx = torch.randint(0, uncovered.numel(), (1,)).item()
            return int(uncovered[idx].item())
        else:
            raise ValueError(
                f"Unknown seed_mode={seed_mode}. "
                f"Use 'degree_iter' or 'random_iter'."
            )

    while len(raw_subgraphs) < max_subgraphs:
        seed_idx = pick_next_seed()
        if seed_idx is None:
            break

        s_int = int(seed_idx)

        # PPR
        if solver == "push":
            p_cpu, _ = local_push_ppr(adj, deg_cpu, s_int, alpha=alpha, eps=eps)
            p = p_cpu.to(device_graph)
        else:
            p, _ = local_power_ppr(edge_index_ppr, deg_ppr, s_int,
                                   alpha=alpha, num_iters=ppr_num_iters, tol=ppr_tol)
            p_cpu = p.cpu()

        # hybrid sweep
        core_cpu = hybrid_sweep_cut(adj, deg_cpu, p_cpu,
                                    vol_min_frac=vol_min_frac,
                                    vol_max_frac=vol_max_frac)
        if core_cpu.numel() == 0:
            covered[s_int] = True
            continue
        core = core_cpu.to(device_graph)

        # halo
        nodes, depths = bfs_khop(data, core, k=khop)
        nodes = nodes.to(device_graph)
        depths = depths.to(device_graph)
        if nodes.numel() == 0:
            covered[s_int] = True
            continue

        edges = edges_from_nodes(data, nodes).to(device_graph)
        # covered[nodes.cpu()] = True
        if coverage_mode == "core":
            covered[core_cpu] = True
        elif coverage_mode == "core_halo":
            covered[nodes.cpu()] = True
        else:
            raise ValueError("coverage_mode must be 'core' or 'core_halo'")

        # markings
        is_seed = (nodes == s_int).float()
        pvals = p_cpu[nodes]
        node_mark = torch.stack([is_seed, pvals, depths.float()], dim=-1)

        depth_root = depth_root_bfs(edge_index_cpu, nodes, edges, s_int, device_graph, max_k=5)
  
        
        node_marking_dict = {
            "is_root": is_seed,
            "is_core": (nodes.unsqueeze(1) == core.unsqueeze(0)).any(dim=1).float(),
            "depth_root": depth_root,
            "depth_core": depths,
            "ppr_value": pvals,
            "ppr_normalized": pvals / (pvals.sum() + 1e-15),
            "ppr_degree_normalized": pvals / (deg_cpu[nodes.cpu()].clamp_min(1.0).to(device_graph)),
        }

        if use_edge_markings:
            u, v = data.edge_index
            e_u = u[edges]
            e_v = v[edges]
            p_u = p[e_u]
            p_v = p[e_v]
            e_mark = torch.stack(
                [torch.minimum(p_u, p_v), torch.maximum(p_u, p_v)], dim=-1
            )
        else:
            e_mark = None

        raw_subgraphs.append(
            {
                "nodes": nodes,
                "edges": edges,
                "node_markings": node_mark,
                "edge_markings": e_mark,
                "node_marking_dict": node_marking_dict,
                "is_root": is_seed,
                "is_core": node_marking_dict["is_core"],
                "score": node_marking_dict["ppr_value"],
            }
        )

    # 6) rescue phase (unchanged; uses graph device)
    uncovered = (~covered).nonzero(as_tuple=False).view(-1)
    if uncovered.numel() > 0:
        if max_rescue is None:
            max_rescue = uncovered.numel()
        rescued = 0
        print("Rescue phase activated for Iterative.")
        for u0 in uncovered.tolist():
            if rescued >= max_rescue:
                break
            s_int = int(u0)
            core = torch.tensor([s_int], dtype=torch.long, device=device_graph)
            nodes, depths = bfs_khop(data, core, k=1)
            nodes = nodes.to(device_graph)
            depths = depths.to(device_graph)
            edges = edges_from_nodes(data, nodes).to(device_graph)
            covered[nodes.cpu()] = True

            # compute markings
            is_seed = (nodes == s_int).float()
            pvals = torch.zeros_like(is_seed)
            node_mark = torch.stack([is_seed, pvals, depths.float()], dim=-1)

            # NOTE: max_k is set to 5
            depth_root = depth_root_bfs(edge_index_cpu, nodes, edges, s_int, device_graph, max_k=5)
            
            node_marking_dict = {
            "is_root": is_seed,
            "is_core": (nodes.unsqueeze(1) == core.unsqueeze(0)).any(dim=1).float(),
            "depth_root": depth_root,
            "depth_core": depths,
            "ppr_value": pvals,
            "ppr_normalized": pvals,
            "ppr_degree_normalized": pvals,
            }


            if use_edge_markings:
                u, v = data.edge_index
                e_u = u[edges]
                e_v = v[edges]
                zeros = torch.zeros_like(e_u, dtype=torch.float32)
                e_mark = torch.stack([zeros, zeros], dim=-1)
            else:
                e_mark = None

            raw_subgraphs.append(
                {
                    "nodes": nodes,
                    "edges": edges,
                    "node_markings": node_mark,
                    "edge_markings": e_mark,
                    "node_marking_dict": node_marking_dict,
                    "is_root": is_seed,
                    "is_core": node_marking_dict["is_core"],
                    "score": node_marking_dict["ppr_value"],
                }
            )
            rescued += 1

    # 7) Final formatting
    return raw_subgraphs

def acl_push_auto_policy(
    data: Data,
    mode: str = "standard",
    seed_mode: str = "farthest",
    max_seeds: Optional[int] = None,
    max_subgraphs: Optional[int] = None,
    coverage_mode: str = "core_halo",
    alpha: float = 0.3,
    eps: float = 1e-4,
    khop: int = 1,
    vol_min_frac: float = 0.05,
    vol_max_frac: float = 0.20,
    ppr_num_iters: int = 5,
    ppr_tol: float = 1e-6,
    max_rescue: Optional[int] = None,
    use_node_markings: bool = True,
    use_edge_markings: bool = True,
    ppr_use_cuda: bool = False,    # NEW
    chosen_solver: str = None,
) -> Dict[str, List]:

    mode = mode.lower()
    seed_mode = seed_mode.lower()

    if mode == "standard":
        return acl_policy(
            data=data,
            seed_mode=seed_mode,
            max_seeds=max_seeds,
            alpha=alpha,
            eps=eps,
            khop=khop,
            vol_min_frac=vol_min_frac,
            vol_max_frac=vol_max_frac,
            ppr_num_iters=ppr_num_iters,
            ppr_tol=ppr_tol,
            max_rescue=max_rescue,
            use_node_markings=use_node_markings,
            use_edge_markings=use_edge_markings,
            ppr_use_cuda=ppr_use_cuda,
            chosen_solver=chosen_solver,
        )

    elif mode == "iterative":
        if seed_mode == "degree":
            iter_seed_mode = "degree_iter"
        elif seed_mode == "random":
            iter_seed_mode = "random_iter"
        else:
            raise ValueError(
                f"seed_mode='{seed_mode}' is not supported in mode='iterative'. "
                f"Use 'degree' or 'random' for iterative seeding."
            )

        return acl_policy_iterative(
            data=data,
            seed_mode=iter_seed_mode,
            max_subgraphs=max_subgraphs,
            coverage_mode=coverage_mode,
            alpha=alpha,
            eps=eps,
            khop=khop,
            vol_min_frac=vol_min_frac,
            vol_max_frac=vol_max_frac,
            ppr_num_iters=ppr_num_iters,
            ppr_tol=ppr_tol,
            max_rescue=max_rescue,
            use_node_markings=use_node_markings,
            use_edge_markings=use_edge_markings,
            ppr_use_cuda=ppr_use_cuda,
            chosen_solver=chosen_solver,
        )

    else:
        raise ValueError(
            f"Unknown mode='{mode}'. Use 'standard' or 'iterative'."
        )


def seeded_tesselating_egonet(
    data: Data,
    seed_mode: str = "degree_iter",   # "degree_iter" or "random_iter"
    khop: int = 2,
    pad_hop: int = 1,
    coverage_mode: str = "core_halo", # "core" or "core_halo"
    use_node_markings: bool = True,
    use_edge_markings: bool = True,
) -> List[Dict[str, Any]]:
    """
    Seeded local neighborhood extraction by iterative k-hop egonets.

    - Pick seeds iteratively among *currently uncovered* nodes
      (by highest degree or randomly, depending on `seed_mode`).
    - For each seed s:
        core_nodes  = k-hop egonet around s
        nodes, depth_core = pad_hop expansion around core_nodes
        edges       = induced edges on `nodes`
      (core = khop shell; halo = pad_hop around that core)
    - Mark coverage depending on `coverage_mode`:
        "core"      -> only core_nodes count as covered
        "core_halo" -> nodes (core + halo) count as covered
    - Repeat until the graph is fully covered (under the chosen coverage).

    Returns:
        raw_subgraphs: List[dict] with keys:
          - "nodes": LongTensor
          - "edges": LongTensor
          - "node_markings": Tensor [num_nodes_sub, 3] (for legacy use)
          - "edge_markings": Tensor or None
          - "node_marking_dict": dict of named features
          - "is_root": Tensor [num_nodes_sub] (same as "is_root" channel)
          - "is_core": Tensor [num_nodes_sub]
          - "score":  Tensor [num_nodes_sub]  (here: zeros, PPR placeholder)
    """
    device_graph = data.edge_index.device
    N = data.num_nodes
    edge_index_cpu = data.edge_index.cpu()
    u, v = edge_index_cpu

    # ------------------------------------------------------------------
    # 1) Degree (on CPU) for seed selection
    # ------------------------------------------------------------------
    deg_cpu = torch.bincount(
        torch.cat([u, v], dim=0),
        minlength=N
    ).float()

    # ------------------------------------------------------------------
    # 2) Coverage mask (on CPU)
    # ------------------------------------------------------------------
    covered = torch.zeros(N, dtype=torch.bool)  # "covered" depends on coverage_mode

    # ------------------------------------------------------------------
    # 3) Seed picker
    # ------------------------------------------------------------------
    def pick_next_seed() -> Optional[int]:
        uncovered = (~covered).nonzero(as_tuple=False).view(-1)
        if uncovered.numel() == 0:
            return None

        if seed_mode == "degree_iter":
            cand_deg = deg_cpu[uncovered]
            return int(uncovered[torch.argmax(cand_deg)].item())
        elif seed_mode == "random_iter":
            idx = torch.randint(0, uncovered.numel(), (1,)).item()
            return int(uncovered[idx].item())
        else:
            raise ValueError(
                f"Unknown seed_mode={seed_mode}. "
                f"Use 'degree_iter' or 'random_iter'."
            )

    raw_subgraphs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 4) Main iterative loop: keep adding egonets until coverage is done
    # ------------------------------------------------------------------
    while True:
        seed_idx = pick_next_seed()
        if seed_idx is None:
            break

        s_int = int(seed_idx)

        # ---- core: khop egonet around the seed
        seed_tensor = torch.tensor([s_int], dtype=torch.long, device=device_graph)
        core_nodes, _ = bfs_khop(data, seed_tensor, k=khop)
        core_nodes = core_nodes.to(device_graph)
        core_cpu = core_nodes.cpu()

        # ---- pad_hop halo around the core
        nodes, depth_core = bfs_khop(data, core_nodes, k=pad_hop)
        nodes = nodes.to(device_graph)
        depth_core = depth_core.to(device_graph)

        if nodes.numel() == 0:
            # mark seed as covered to avoid infinite loop
            covered[s_int] = True
            continue

        # induced edges on nodes
        edges = edges_from_nodes(data, nodes).to(device_graph)

        # ---- update coverage
        if coverage_mode == "core":
            covered[core_cpu] = True
        elif coverage_mode == "core_halo":
            covered[nodes.cpu()] = True
        else:
            raise ValueError("coverage_mode must be 'core' or 'core_halo'")

        # ------------------------------------------------------------------
        # 5) Node markings (dict + legacy stacked tensor)
        # ------------------------------------------------------------------
        # is_root (seed indicator)
        is_seed = (nodes == s_int).float()

        # is_core (membership in khop egonet)
        is_core = (nodes.unsqueeze(1) == core_nodes.unsqueeze(0)).any(dim=1).float()

        # depth_root: BFS distance from seed (truncated)
        depth_root = depth_root_bfs(
            edge_index_cpu=edge_index_cpu,
            nodes=nodes,
            edges=edges,
            s_int=s_int,
            device_graph=device_graph,
            max_k=khop + pad_hop,  # or any fixed cutoff like 5
        )

        # No PPR here; keep placeholders for API compatibility
        pvals = torch.zeros_like(is_seed)
        pvals_norm = pvals  # remains zero
        pvals_deg_norm = pvals  # remains zero

        node_marking_dict = {
            "is_root": is_seed,
            "is_core": is_core,
            "depth_root": depth_root,
            "depth_core": depth_core,
            "ppr_value": pvals,
            "ppr_normalized": pvals_norm,
            "ppr_degree_normalized": pvals_deg_norm,
        }

        # Legacy stacked tensor (if you still use it somewhere):
        # [is_root, "score" (here 0), depth_core]
        node_mark = torch.stack(
            [is_seed, pvals, depth_core.float()],
            dim=-1,
        ) if use_node_markings else None

        # ------------------------------------------------------------------
        # 6) Edge markings (simple core-edge indicator placeholder)
        # ------------------------------------------------------------------
        if use_edge_markings:
            u_full, v_full = data.edge_index
            e_u = u_full[edges]
            e_v = v_full[edges]

            core_mask_global = torch.zeros(N, dtype=torch.bool, device=device_graph)
            core_mask_global[core_nodes] = True
            is_core_edge = (core_mask_global[e_u] & core_mask_global[e_v]).float()

            # 2-channel placeholder, similar shape as ACL edge markings
            e_mark = torch.stack(
                [is_core_edge, torch.zeros_like(is_core_edge)],
                dim=-1,
            )
        else:
            e_mark = None

        # ------------------------------------------------------------------
        # 7) Store subgraph
        # ------------------------------------------------------------------
        raw_subgraphs.append(
            {
                "nodes": nodes,
                "edges": edges,
                "node_markings": node_mark,
                "edge_markings": e_mark,
                "node_marking_dict": node_marking_dict,
                "is_root": is_seed,
                "is_core": is_core,
                "score": pvals,  # placeholder to match ACL structure
            }
        )

    return raw_subgraphs
