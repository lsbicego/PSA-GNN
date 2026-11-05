from __future__ import annotations

import copy
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import from_networkx
from tqdm.auto import tqdm


def _macro_pair_prob_matrix_ring(M: int, p_out_base: float, delta: float) -> np.ndarray:
    """Build between-macro probabilities for a ring topology."""
    if M < 3:
        raise ValueError("num_macros must be >= 3")

    n_total = M * (M - 1) // 2
    n_strong = M

    p_high = float(np.clip(p_out_base + delta, 0.0, 1.0))
    if n_total == n_strong:
        p_low = p_high
    else:
        p_low = (n_total * p_out_base - n_strong * p_high) / (n_total - n_strong)
    p_low = float(np.clip(p_low, 0.0, 1.0))

    probs = np.full((M, M), p_low, dtype=float)
    np.fill_diagonal(probs, 0.0)
    for m in range(M):
        n = (m + 1) % M
        probs[m, n] = p_high
        probs[n, m] = p_high
    return probs


def _macro_pair_prob_matrix_bipartite(M: int, p_out_base: float, delta: float) -> np.ndarray:
    """Build between-macro probabilities for a bipartite-like topology."""
    if M < 2:
        raise ValueError("num_macros must be >= 2")

    m1 = M // 2
    m2 = M - m1
    n_total = M * (M - 1) // 2
    n_cross = m1 * m2
    n_same = n_total - n_cross

    p_cross = float(np.clip(p_out_base + delta, 0.0, 1.0))
    if n_same == 0:
        p_same = p_cross
    else:
        p_same = (n_total * p_out_base - n_cross * p_cross) / n_same
    p_same = float(np.clip(p_same, 0.0, 1.0))

    probs = np.zeros((M, M), dtype=float)
    for i in range(M):
        for j in range(i + 1, M):
            same_part = ((i < m1) and (j < m1)) or ((i >= m1) and (j >= m1))
            probs[i, j] = probs[j, i] = p_same if same_part else p_cross
    return probs


def generate_hsbm_macro_topology_graph(
    num_macros: int = 6,
    micros_per_macro: int = 4,
    nodes_per_micro: int = 10,
    p_in_micro: float = 0.6,
    p_in_macro: float = 0.2,
    p_out_base: float = 0.03,
    delta_macro: float = 0.015,
    seed: Optional[int] = None,
    class_type: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Generate one graph for hSBM Variant C.

    Class 0 uses ring-like macro connectivity; class 1 uses bipartite-like macro connectivity.
    """
    if num_macros < 2:
        raise ValueError("num_macros must be >= 2")
    if micros_per_macro < 1:
        raise ValueError("micros_per_macro must be >= 1")
    if nodes_per_micro < 1:
        raise ValueError("nodes_per_micro must be >= 1")

    rng = np.random.default_rng(seed)
    class_value = int(rng.integers(0, 2)) if class_type is None else int(class_type)
    if class_value not in (0, 1):
        raise ValueError("class_type must be 0, 1, or None")

    M = int(num_macros)
    K = int(micros_per_macro)
    S = int(nodes_per_micro)
    N = M * K * S

    if class_value == 0:
        macro_topology = "ring"
        probs_macro = _macro_pair_prob_matrix_ring(M=M, p_out_base=p_out_base, delta=delta_macro)
    else:
        macro_topology = "bipartite"
        probs_macro = _macro_pair_prob_matrix_bipartite(M=M, p_out_base=p_out_base, delta=delta_macro)

    G = nx.Graph()
    micro_nodes: List[List[int]] = []
    micro_to_macro: List[int] = []

    node_idx = 0
    for macro in range(M):
        for micro_in_macro in range(K):
            micro_global = macro * K + micro_in_macro
            nodes: List[int] = []
            for _ in range(S):
                n = node_idx
                node_idx += 1
                G.add_node(
                    n,
                    macro_id=macro,
                    micro_global_id=micro_global,
                    micro_in_macro_id=micro_in_macro,
                )
                nodes.append(n)
            micro_nodes.append(nodes)
            micro_to_macro.append(macro)

    for nodes in micro_nodes:
        s = len(nodes)
        if p_in_micro <= 0.0 or s < 2:
            continue
        upper = np.triu(rng.random((s, s)) < p_in_micro, k=1)
        ii, jj = np.where(upper)
        G.add_edges_from((nodes[i], nodes[j]) for i, j in zip(ii, jj))

    num_micros = len(micro_nodes)
    for i in range(num_micros):
        for j in range(i + 1, num_micros):
            macro_i = micro_to_macro[i]
            macro_j = micro_to_macro[j]
            prob = p_in_macro if macro_i == macro_j else float(probs_macro[macro_i, macro_j])
            if prob <= 0.0:
                continue
            nodes_i = micro_nodes[i]
            nodes_j = micro_nodes[j]
            edge_mask = rng.random((len(nodes_i), len(nodes_j))) < prob
            ii, jj = np.where(edge_mask)
            G.add_edges_from((nodes_i[a], nodes_j[b]) for a, b in zip(ii, jj))

    meta: Dict[str, Any] = {
        "num_nodes": N,
        "num_macros": M,
        "micros_per_macro": K,
        "nodes_per_micro": S,
        "class_type": class_value,
        "label": class_value,
        "y": class_value,
        "macro_topology": macro_topology,
        "p_in_micro": float(p_in_micro),
        "p_in_macro": float(p_in_macro),
        "p_out_base": float(p_out_base),
        "delta_macro": float(delta_macro),
    }
    return G, meta


def _sample_mks(
    rng: np.random.Generator,
    target_num_nodes: int = 200,
    M_choices: Sequence[int] = (5, 6, 7, 8),
    K_choices: Sequence[int] = (3, 4, 5),
    S_min: int = 8,
    S_max: int = 16,
) -> Tuple[int, int, int, int]:
    """Sample (M, K, S, N) with N close to target_num_nodes."""
    M = int(rng.choice(M_choices))
    K = int(rng.choice(K_choices))
    S_raw = int(round(target_num_nodes / (M * K)))
    S = int(np.clip(S_raw, S_min, S_max))
    S += int(rng.integers(-1, 2))
    S = int(np.clip(S, S_min, S_max))
    N = M * K * S
    return M, K, S, N


def generate_hsbm_macro_topology_variable_dataset(
    num_graphs: int,
    target_num_nodes: int = 200,
    M_choices: Sequence[int] = (5, 6, 7, 8),
    K_choices: Sequence[int] = (3, 4, 5),
    p_in_micro: float = 0.35,
    p_in_macro: float = 0.12,
    p_out_base: float = 0.01,
    delta_macro: float = 0.005,
    seed: Optional[int] = None,
) -> Tuple[List[nx.Graph], List[Dict[str, Any]]]:
    """Generate variable-size graphs using hSBM Variant C."""
    rng = np.random.default_rng(seed)
    graphs: List[nx.Graph] = []
    metas: List[Dict[str, Any]] = []

    for i in range(num_graphs):
        M, K, S, sampled_num_nodes = _sample_mks(
            rng=rng,
            target_num_nodes=target_num_nodes,
            M_choices=M_choices,
            K_choices=K_choices,
            S_min=8,
            S_max=16,
        )
        graph_seed = int(rng.integers(0, 10**9))
        class_type = int(rng.integers(0, 2))
        G, meta = generate_hsbm_macro_topology_graph(
            num_macros=M,
            micros_per_macro=K,
            nodes_per_micro=S,
            p_in_micro=p_in_micro,
            p_in_macro=p_in_macro,
            p_out_base=p_out_base,
            delta_macro=delta_macro,
            seed=graph_seed,
            class_type=class_type,
        )
        meta["graph_id"] = i
        meta["sampled_num_nodes"] = sampled_num_nodes
        graphs.append(G)
        metas.append(meta)
    return graphs, metas


def _balanced_sizes(n: int, M: int) -> List[int]:
    """Near-equal block sizes summing to n."""
    base = n // M
    rem = n - base * M
    sizes = [base] * M
    for i in range(rem):
        sizes[i] += 1
    return sizes


def _barbell_sizes(n: int, M: int, ratio: float = 4.0) -> List[int]:
    """Two large blocks plus many smaller blocks, summing to n."""
    if M < 4:
        raise ValueError("M must be >= 4 for barbell sizes")
    s = max(3, int(round(n / (2.0 * ratio + (M - 2)))))
    L = int(round(ratio * s))
    sizes = [L, L] + [s] * (M - 2)

    diff = n - sum(sizes)
    i = 2
    while diff > 0:
        sizes[i % M] += 1
        diff -= 1
        i += 1

    while diff < 0:
        j = 0
        while diff < 0 and j < M:
            if sizes[j] > 3:
                sizes[j] -= 1
                diff += 1
            j += 1
        if j == M and diff < 0:
            break

    sizes.sort(reverse=True)
    return sizes


def _size_shape_label(sizes: List[int], mode: str = "two_big_vs_not") -> int:
    """Map block-size shape to a class label."""
    s = np.array(sorted(sizes))
    threshold = float(np.median(s) * 1.5)
    num_big = int(np.sum(s >= threshold))
    if mode == "two_big_vs_not":
        return 1 if num_big == 2 else 0
    if mode == "parity_big":
        return num_big % 2
    raise ValueError(f"Unknown label_mode: {mode}")


@dataclass
class DegreeMatchedSBMParams:
    d_tot: float
    p_in_min: float = 1e-4
    p_in_max: float = 0.95


def _block_pin_for_degree_matching(
    n: int,
    sizes: List[int],
    p_out: float,
    params: DegreeMatchedSBMParams,
) -> List[float]:
    """Choose per-block p_in to approximately match expected degree target."""
    pins: List[float] = []
    for s in sizes:
        if s <= 1:
            pins.append(params.p_in_min)
            continue
        d_out = (n - s) * p_out
        d_in = params.d_tot - d_out
        p_in = d_in / max(1.0, (s - 1))
        pins.append(float(np.clip(p_in, params.p_in_min, params.p_in_max)))
    return pins


def _expected_mean_degree(n: int, sizes: List[int], pins: List[float], p_out: float) -> float:
    """Expected mean degree under the degree-matched SBM parameterization."""
    s = np.array(sizes, dtype=float)
    p = np.array(pins, dtype=float)
    deg_by_block = (s - 1.0) * p + (n - s) * p_out
    return float(np.sum(deg_by_block * s) / np.sum(s))


def _tune_p_out_for_target_mean_degree(
    n: int,
    sizes: List[int],
    d_target: float,
    params: DegreeMatchedSBMParams,
    p_out_lo: float = 1e-4,
    p_out_hi: float = 0.05,
    iters: int = 25,
) -> Tuple[float, List[float], float]:
    """Binary-search p_out to match target expected mean degree."""
    lo = float(p_out_lo)
    hi = float(p_out_hi)
    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        pins = _block_pin_for_degree_matching(n=n, sizes=sizes, p_out=mid, params=params)
        mean_deg = _expected_mean_degree(n=n, sizes=sizes, pins=pins, p_out=mid)
        best = (mid, pins, mean_deg)
        if mean_deg < d_target:
            lo = mid
        else:
            hi = mid
    if best is None:
        raise RuntimeError("Failed to tune p_out")
    return best


def _sample_sbm_graph(sizes: List[int], pins: List[float], p_out: float, seed: int) -> nx.Graph:
    """Sample an undirected SBM with per-block p_in and global p_out."""
    rng = np.random.default_rng(seed)
    n = int(sum(sizes))

    block_id = np.empty(n, dtype=np.int64)
    blocks: List[np.ndarray] = []
    start = 0
    for b, s in enumerate(sizes):
        idx = np.arange(start, start + s, dtype=np.int64)
        block_id[idx] = b
        blocks.append(idx)
        start += s

    G = nx.Graph()
    G.add_nodes_from(range(n))
    nx.set_node_attributes(G, {i: int(block_id[i]) for i in range(n)}, "block_id")

    for b, idx in enumerate(blocks):
        s = len(idx)
        if s <= 1:
            continue
        p = float(pins[b])
        upper = np.triu(rng.random((s, s)) < p, k=1)
        ii, jj = np.where(upper)
        G.add_edges_from((int(idx[i]), int(idx[j])) for i, j in zip(ii, jj))

    B = len(blocks)
    for a in range(B):
        ia = blocks[a]
        for b in range(a + 1, B):
            ib = blocks[b]
            if len(ia) == 0 or len(ib) == 0:
                continue
            mat = rng.random((len(ia), len(ib))) < p_out
            ii, jj = np.where(mat)
            G.add_edges_from((int(ia[i]), int(ib[j])) for i, j in zip(ii, jj))
    return G


def generate_degree_matched_granularity_dataset_v2(
    num_graphs: int = 10_000,
    num_nodes: int = 200,
    M_choices: Tuple[int, ...] = (6, 7, 8),
    p_out_base: float = 0.01,
    d_tot: float = 14.5,
    barbell_ratio: float = 4.0,
    label_mode: str = "two_big_vs_not",
    balance_M_across_classes: bool = True,
    tune_p_out_per_graph: bool = False,
    p_out_search_range: Tuple[float, float] = (1e-4, 0.05),
    mean_deg_target: Optional[float] = None,
    seed: int = 0,
) -> Tuple[List[nx.Graph], List[Dict[str, Any]]]:
    """Generate the degree-matched size-granularity dataset."""
    rng = random.Random(seed)
    params = DegreeMatchedSBMParams(d_tot=float(d_tot))

    if mean_deg_target is None:
        M0 = int(np.median(list(M_choices)))
        sizes0 = _balanced_sizes(num_nodes, M0)
        pins0 = _block_pin_for_degree_matching(num_nodes, sizes0, p_out_base, params)
        mean_deg_target = _expected_mean_degree(num_nodes, sizes0, pins0, p_out_base)

    graphs: List[nx.Graph] = []
    metas: List[Dict[str, Any]] = []

    M_list = list(M_choices)
    n_per_class = num_graphs // 2

    if balance_M_across_classes:
        pairs: List[Tuple[int, int]] = []
        per_M = n_per_class // len(M_list)
        remainder = n_per_class - per_M * len(M_list)
        for y in [0, 1]:
            counts = {M: per_M for M in M_list}
            shuffled_M = M_list.copy()
            rng.shuffle(shuffled_M)
            for i in range(remainder):
                counts[shuffled_M[i % len(M_list)]] += 1
            for M, c in counts.items():
                pairs.extend([(y, M)] * c)
        while len(pairs) < num_graphs:
            pairs.append((rng.randint(0, 1), rng.choice(M_list)))
        rng.shuffle(pairs)
        yM_pairs = pairs[:num_graphs]
    else:
        labels = [0] * n_per_class + [1] * (num_graphs - n_per_class)
        rng.shuffle(labels)
        yM_pairs = [(labels[i], rng.choice(M_list)) for i in range(num_graphs)]

    for i, (y_requested, M) in enumerate(yM_pairs):
        sizes = (
            _balanced_sizes(num_nodes, M)
            if y_requested == 0
            else _barbell_sizes(num_nodes, M, ratio=barbell_ratio)
        )
        y = int(_size_shape_label(sizes, mode=label_mode))

        if tune_p_out_per_graph:
            p_lo, p_hi = p_out_search_range
            p_out, pins, mean_deg = _tune_p_out_for_target_mean_degree(
                n=num_nodes,
                sizes=sizes,
                d_target=float(mean_deg_target),
                params=params,
                p_out_lo=float(p_lo),
                p_out_hi=float(p_hi),
                iters=25,
            )
        else:
            p_out = float(p_out_base)
            pins = _block_pin_for_degree_matching(num_nodes, sizes, p_out, params)
            mean_deg = _expected_mean_degree(num_nodes, sizes, pins, p_out)

        graph = _sample_sbm_graph(
            sizes=sizes,
            pins=pins,
            p_out=p_out,
            seed=seed * 1_000_003 + i,
        )
        graphs.append(graph)
        metas.append(
            {
                "y": y,
                "label": y,
                "y_requested": int(y_requested),
                "n": int(num_nodes),
                "M": int(M),
                "block_sizes": sizes,
                "p_in_blocks": pins,
                "p_out": float(p_out),
                "d_tot_target": float(d_tot),
                "mean_deg_target": float(mean_deg_target),
                "mean_deg_expected": float(mean_deg),
                "barbell_ratio": float(barbell_ratio),
                "label_mode": label_mode,
                "balance_M_across_classes": bool(balance_M_across_classes),
                "tune_p_out_per_graph": bool(tune_p_out_per_graph),
            }
        )
    return graphs, metas


def generate_degree_matched_granularity_graph(
    num_nodes: int = 200,
    M: int = 7,
    p_out_base: float = 0.01,
    d_tot: float = 14.5,
    barbell_ratio: float = 4.0,
    label_mode: str = "two_big_vs_not",
    class_type: Optional[int] = None,
    tune_p_out_per_graph: bool = False,
    p_out_search_range: Tuple[float, float] = (1e-4, 0.05),
    mean_deg_target: Optional[float] = None,
    seed: Optional[int] = None,
) -> Tuple[nx.Graph, Dict[str, Any]]:
    """
    Generate a single degree-matched granularity graph.

    class_type:
      - 0: balanced block-size template
      - 1: barbell block-size template
    """
    if M < 4:
        raise ValueError("M must be >= 4")
    rng = np.random.default_rng(seed)
    class_value = int(rng.integers(0, 2)) if class_type is None else int(class_type)
    if class_value not in (0, 1):
        raise ValueError("class_type must be 0, 1, or None")

    params = DegreeMatchedSBMParams(d_tot=float(d_tot))
    if mean_deg_target is None:
        sizes0 = _balanced_sizes(num_nodes, M)
        pins0 = _block_pin_for_degree_matching(num_nodes, sizes0, p_out_base, params)
        mean_deg_target = _expected_mean_degree(num_nodes, sizes0, pins0, p_out_base)

    if class_value == 0:
        sizes = _balanced_sizes(num_nodes, M)
    else:
        sizes = _barbell_sizes(num_nodes, M, ratio=barbell_ratio)

    y = int(_size_shape_label(sizes, mode=label_mode))

    if tune_p_out_per_graph:
        p_lo, p_hi = p_out_search_range
        p_out, pins, mean_deg = _tune_p_out_for_target_mean_degree(
            n=num_nodes,
            sizes=sizes,
            d_target=float(mean_deg_target),
            params=params,
            p_out_lo=float(p_lo),
            p_out_hi=float(p_hi),
            iters=25,
        )
    else:
        p_out = float(p_out_base)
        pins = _block_pin_for_degree_matching(num_nodes, sizes, p_out, params)
        mean_deg = _expected_mean_degree(num_nodes, sizes, pins, p_out)

    sample_seed = int(rng.integers(0, 10**9))
    graph = _sample_sbm_graph(sizes=sizes, pins=pins, p_out=p_out, seed=sample_seed)
    meta = {
        "y": y,
        "label": y,
        "class_type": int(class_value),
        "n": int(num_nodes),
        "M": int(M),
        "block_sizes": sizes,
        "p_in_blocks": pins,
        "p_out": float(p_out),
        "d_tot_target": float(d_tot),
        "mean_deg_target": float(mean_deg_target),
        "mean_deg_expected": float(mean_deg),
        "barbell_ratio": float(barbell_ratio),
        "label_mode": label_mode,
        "tune_p_out_per_graph": bool(tune_p_out_per_graph),
    }
    return graph, meta


def _infer_label(meta: Dict[str, Any], label_key: Optional[str]) -> int:
    if label_key is not None:
        if label_key not in meta:
            raise KeyError(f"Label key '{label_key}' not found in metadata.")
        return int(meta[label_key])

    has_y = "y" in meta
    has_label = "label" in meta
    if has_y and has_label and int(meta["y"]) != int(meta["label"]):
        raise ValueError("Metadata has both 'y' and 'label' with different values.")
    if has_y:
        return int(meta["y"])
    if has_label:
        return int(meta["label"])
    raise KeyError("Metadata must contain 'y' or 'label', or pass label_key explicitly.")


######## SAVING ##########


def save_graph_dataset_to_disk(
    graphs: Sequence[nx.Graph],
    metas: Sequence[Dict[str, Any]],
    directory: str,
    label_key: Optional[str] = None,
    random_node_features: bool = False,
    node_feature_dim: int = 1,
    include_meta_attributes: bool = True,
    strip_graph_attributes: bool = True,
    show_progress: bool = True,
) -> None:
    """
    Save NetworkX graph datasets as PyG `.pt` files.

    Labels are resolved from metadata with support for both keys: `y` and `label`.
    """
    if len(graphs) != len(metas):
        raise ValueError(f"graphs ({len(graphs)}) and metas ({len(metas)}) must have same length.")
    if node_feature_dim < 1:
        raise ValueError("node_feature_dim must be >= 1")

    os.makedirs(directory, exist_ok=True)
    iterator: Iterable[Tuple[nx.Graph, Dict[str, Any]]] = zip(graphs, metas)
    if show_progress:
        iterator = tqdm(iterator, total=len(graphs), desc="Saving graphs")

    for i, (G, meta) in enumerate(iterator):
        H = copy.deepcopy(G)
        if strip_graph_attributes:
            for n in H.nodes():
                H.nodes[n].clear()
            for _, _, data in H.edges(data=True):
                data.clear()

        data = from_networkx(H)
        label = _infer_label(meta=meta, label_key=label_key)
        data.y = torch.tensor([label], dtype=torch.long)
        if random_node_features:
            data.x = torch.rand((data.num_nodes, node_feature_dim), dtype=torch.float)
        else:
            data.x = torch.ones((data.num_nodes, node_feature_dim), dtype=torch.float)

        if include_meta_attributes:
            reserved_keys = {"x", "y", "edge_index", "num_nodes"}
            for k, v in meta.items():
                if k in reserved_keys:
                    continue
                setattr(data, k, v)

        torch.save(data, os.path.join(directory, f"graph_{i}.pt"))


__all__ = [
    "generate_hsbm_macro_topology_graph",
    "generate_hsbm_macro_topology_variable_dataset",
    "generate_degree_matched_granularity_graph",
    "generate_degree_matched_granularity_dataset_v2",
    "save_graph_dataset_to_disk",
]
