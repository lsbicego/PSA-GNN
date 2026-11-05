"""Optional nested-SBM descriptors for one PyTorch Geometric graph.

This independent implementation follows ``scripts/hierarchy_scoring.py``; the
archival script remains unchanged. Importing this module does not import
graph-tool or perform inference. The descriptors summarize topology, not whether
that topology predicts a graph label.
"""

from __future__ import annotations

import json
import operator
import subprocess
import sys
from typing import Any

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.transforms import BaseTransform

__all__ = ["compute_hierarchy_stats", "NestedSBMStats"]


def _topology(data: Data) -> tuple[int, list[tuple[int, int]], int]:
    """Validate homogeneous topology and retain isolated vertices explicitly."""
    if not isinstance(data, Data) or isinstance(data, Batch):
        raise TypeError("Expected one homogeneous PyG Data graph, not a Batch.")
    if data.num_nodes is None:
        raise ValueError("Set data.num_nodes (including isolated nodes) before fitting.")
    num_nodes = operator.index(data.num_nodes)
    if num_nodes < 1:
        raise ValueError("Nested-SBM characterization requires at least one node.")
    edge_index = data.edge_index
    if edge_index is None:
        return num_nodes, [], 0
    if not torch.is_tensor(edge_index) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("data.edge_index must be an integer tensor of shape [2, E].")
    if edge_index.dtype not in (torch.int32, torch.int64):
        raise ValueError("data.edge_index must contain integer node indices (int32 or int64).")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise ValueError("edge_index contains indices outside [0, data.num_nodes).")
    # Match the archival canonicalization: sorted, unique, undirected pairs.
    pairs = sorted({tuple(sorted((int(u), int(v))))
                    for u, v in edge_index.t().detach().cpu().tolist()})
    removed_self_loops = sum(u == v for u, v in pairs)
    return num_nodes, [(u, v) for u, v in pairs if u != v], removed_self_loops


# Only JSON topology crosses this process boundary. Keeping torch out of the
# native worker avoids incompatible OpenMP/native-library initialization in
# environments that provide both torch and graph-tool (including our tested
# macOS conda environment). There is no shell or unpickling here.
_NATIVE_WORKER = r"""
import json
import sys
try:
    import graph_tool as gt
    from graph_tool.all import Graph
    from graph_tool.inference import minimize_blockmodel_dl, minimize_nested_blockmodel_dl
except (ImportError, OSError) as exc:
    print(json.dumps({"error": "missing_dependency", "message": str(exc)}))
    sys.exit(2)
import numpy as np
payload = json.load(sys.stdin)
if gt.openmp_enabled():
    gt.openmp_set_num_threads(payload["num_threads"])
if payload["seed"] is not None:
    gt.seed_rng(payload["seed"])
graph = Graph(directed=False)
graph.add_vertex(payload["num_nodes"])
if payload["edges"]:
    graph.add_edge_list(payload["edges"])
# Deliberate archival RNG parity: do not remove the preceding flat fit.
minimize_blockmodel_dl(graph, state_args={"deg_corr": True})
state = minimize_nested_blockmodel_dl(graph, state_args={"deg_corr": True})
counts = [int(np.unique(np.asarray(membership)).size) for membership in state.get_bs()]
print(json.dumps({"counts": counts, "version": str(gt.__version__),
                  "numpy_version": np.__version__, "python_version": sys.version.split()[0]}))
"""


def _fit_native(payload: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-B", "-c", _NATIVE_WORKER],
        input=json.dumps(payload), text=True, capture_output=True, check=False,
    )
    if result.returncode == 2:
        try:
            dependency_error = json.loads(result.stdout).get("error") == "missing_dependency"
        except (ValueError, AttributeError):
            dependency_error = False
        if dependency_error:
            raise ImportError(
                "Hierarchy characterization requires the optional graph-tool package "
                "in the same Python environment. For example, install it with "
                "`conda install -c conda-forge graph-tool`. Dataset loading and "
                "PSA-GNN training do not require graph-tool."
            )
    if result.returncode:
        raise RuntimeError(
            f"graph-tool fitting process exited with status {result.returncode}. "
            f"Check the optional native environment. {result.stderr.strip()[-2000:]}"
        )
    try:
        return json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError("graph-tool fitting returned an invalid result.") from exc


def compute_hierarchy_stats(
    data: Data, *, seed: int | None = None, num_threads: int = 1
) -> dict[str, Any]:
    """Fit the archival degree-corrected SBM procedure to one graph.

    Edges become undirected and unweighted; duplicates and self-loops are
    discarded. All ``data.num_nodes`` vertices are retained. Node/edge features,
    weights and labels do not enter inference, and ``data`` is not modified.

    ``substructure_count`` counts unique finest-level block memberships.
    ``substructure_depth`` is the number of *stored* membership levels, exactly
    matching the archival ``nested_hierarchy_depth_total_levels`` field. Some
    graph-tool versions retain repeated single-block levels: consequently a
    structurally flat graph need not have depth one. ``level_block_counts`` and
    ``nontrivial_level_count`` expose that distinction without silently changing
    the scientific convention. Neither descriptor measures label relevance.

    A flat fit is intentionally performed before the nested fit, as in the
    archival script: removing it changes the inference RNG sequence. There are
    no additional fitting repetitions or MCMC sweeps. Native fitting runs in a
    fresh subprocess using the same Python executable, with JSON topology
    input/output and no pickle. This avoids native-library conflicts when torch
    has already been imported. A supplied seed resets the worker's graph-tool
    RNG; ``None`` uses its default random state. ``num_threads`` controls the
    worker's OpenMP thread count. The parent's RNG and native thread settings
    are unchanged. Exact values may depend on software/platform versions, and
    fitting can be expensive; prefer cached preprocessing to per-epoch transforms.

    Returns a dictionary with integer descriptors, per-level block counts, and
    the settings/version needed to interpret them. Empty graphs are rejected;
    edgeless nonempty graphs are passed to graph-tool without a special metric.
    """
    num_nodes, edges, removed_self_loops = _topology(data)
    num_threads = operator.index(num_threads)
    if num_threads < 1:
        raise ValueError("num_threads must be a positive integer.")
    if seed is not None:
        seed = operator.index(seed)
        if seed < 0:
            raise ValueError("seed must be a nonnegative integer or None.")
    fitted = _fit_native({"num_nodes": num_nodes, "edges": edges,
                          "seed": seed, "num_threads": num_threads})
    counts = fitted["counts"]
    if not counts:
        raise RuntimeError("graph-tool returned no hierarchy levels; no descriptors can be extracted.")
    return {
        "substructure_count": counts[0],
        "substructure_depth": len(counts),
        "level_block_counts": counts,
        "nontrivial_level_count": sum(count > 1 for count in counts),
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "removed_self_loops": removed_self_loops,
        "inference": {
            "graph_tool_version": fitted["version"],
            "numpy_version": fitted["numpy_version"],
            "python_version": fitted["python_version"],
            "execution": "isolated_process",
            "degree_corrected": True,
            "fit_sequence": ["flat", "nested"],
            "depth_convention": "total_stored_levels",
            "seed": seed,
            "num_threads": num_threads,
        },
    }


class NestedSBMStats(BaseTransform):
    """Cache two graph-level descriptors through a PyG ``pre_transform``.

    ``NestedSBMStats(seed=42)(data)`` follows PyG's BaseTransform convention:
    it returns a shallow copy with length-one ``torch.long`` attributes named
    ``substructure_count`` and ``substructure_depth``. Calling ``forward``
    directly mutates the supplied object. Other attributes are preserved.

    A fixed seed initializes each isolated fit independently of dataset
    enumeration. For parity with archival dataset row ``i``, use the per-graph
    function with ``seed=base_seed + i`` instead. Changing the transform/seed
    requires rebuilding the dataset's processed cache, as with other PyG
    pre-transforms. See :func:`compute_hierarchy_stats` for the depth convention.
    """

    def __init__(self, *, seed: int | None = None, num_threads: int = 1) -> None:
        self.seed = seed
        self.num_threads = num_threads

    def forward(self, data: Data) -> Data:
        stats = compute_hierarchy_stats(data, seed=self.seed, num_threads=self.num_threads)
        data.substructure_count = torch.tensor([stats["substructure_count"]], dtype=torch.long)
        data.substructure_depth = torch.tensor([stats["substructure_depth"]], dtype=torch.long)
        return data

    def __repr__(self) -> str:
        return f"{type(self).__name__}(seed={self.seed!r}, num_threads={self.num_threads!r})"
