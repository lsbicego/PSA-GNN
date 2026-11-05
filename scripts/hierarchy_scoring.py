from __future__ import annotations

"""
Standalone hierarchy scoring for graph datasets using graph-tool SBMs.

Usage:
    python scripts/hierarchy_scoring.py --datasets nci1 zinc
    python scripts/hierarchy_scoring.py --datasets nci1 --run-robustness --robustness-graphs 25
    python scripts/hierarchy_scoring.py --datasets hsbm_ladder --force-reload
    python scripts/hierarchy_scoring.py --datasets hsbm_ladder:harder granularity_ladder:easy

Outputs:
    OUT/hierarchy_scoring/<dataset_name>/
        graph_scores.csv
        delta_entropy_hist.png
        delta_entropy_per_edge_hist.png
        delta_entropy_per_node_hist.png
        run_metadata.json

The script composes Hydra configs internally in the same style as the other
standalone scripts, loads datasets through the existing data pipeline, fits a
flat and nested degree-corrected SBM per graph, and optionally runs lightweight
robustness checks on a subset of graphs.
"""

import argparse
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import random
import sys
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig_hierarchy_scoring"))

import hydra
import matplotlib
matplotlib.use("Agg")
try:
    from graph_tool import openmp_enabled, openmp_set_num_threads, seed_rng
    from graph_tool.all import Graph
    from graph_tool.inference import (
        mcmc_equilibrate,
        minimize_blockmodel_dl,
        minimize_nested_blockmodel_dl,
    )
    GRAPH_TOOL_IMPORT_ERROR: Exception | None = None
except ImportError as exc:
    Graph = None
    minimize_blockmodel_dl = None
    minimize_nested_blockmodel_dl = None
    mcmc_equilibrate = None
    seed_rng = None
    openmp_enabled = None
    openmp_set_num_threads = None
    GRAPH_TOOL_IMPORT_ERROR = exc
import numpy as np
import torch
from torch_geometric import transforms as T
from torch_geometric.datasets import ZINC
from hydra.core.global_hydra import GlobalHydra
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.data_fetcher import get_data
from src.data_processing.data_fetcher import get_subgraph_transform
from src.data_processing.data_utils import ReplaceSelfLoops, fetch_transform

warnings.filterwarnings(
    "ignore",
    message="An issue occurred while importing 'torch-sparse'",
    module="torch_geometric.typing",
)
warnings.filterwarnings(
    "ignore",
    message="An issue occurred while importing 'pyg-lib'",
    module="torch_geometric.typing",
)

CONF_DIR = PROJECT_ROOT / "conf"
LOGGER = logging.getLogger("hierarchy_scoring")


LADDER_DIFFICULTIES = {
    "hsbm_ladder": ("easiest", "easier", "easy", "medium", "hard", "harder", "hardest"),
    "granularity_ladder": ("easy", "easier", "medium", "hard", "harder", "hardest"),
}


@dataclass(frozen=True)
class DatasetRequest:
    raw_request: str
    hydra_dataset_key: str
    dataset_name_override: str | None
    dataset_root_override: str | None
    difficulty: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score graph hierarchy by comparing flat and nested degree-corrected SBMs."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help=(
            "Dataset config names from conf/dataset, e.g. nci1 zinc. "
            "Synthetic ladder datasets also accept an explicit difficulty via "
            "hsbm_ladder:<difficulty> or granularity_ladder:<difficulty>."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="OUT/hierarchy_scoring",
        help="Root directory for outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for dataset splitting, sampling, and graph-tool seeding when available.",
    )
    parser.add_argument(
        "--force-reload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override dataset.force_reload in the composed Hydra config. Default: keep dataset config value.",
    )
    parser.add_argument(
        "--graph-tool-threads",
        type=int,
        default=1,
        help="Number of graph-tool OpenMP threads to use. Default: 1.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of worker processes used to score graphs. "
            "Use values > 1 to parallelize graph-tool fits across graphs."
        ),
    )
    parser.add_argument(
        "--max-graphs",
        type=int,
        default=None,
        help="Optional cap on the number of graphs processed per dataset.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=40,
        help="Number of bins for the score histograms.",
    )
    parser.add_argument(
        "--run-robustness",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run held-out edge and stability checks on a subset of graphs.",
    )
    parser.add_argument(
        "--robustness-graphs",
        type=int,
        default=10,
        help="How many graphs per dataset should receive robustness checks.",
    )
    parser.add_argument(
        "--heldout-edge-fraction",
        type=float,
        default=0.1,
        help="Fraction of observed edges to hold out for edge prediction.",
    )
    parser.add_argument(
        "--heldout-max-pairs",
        type=int,
        default=64,
        help="Maximum number of held-out positive edges and matched negative edges.",
    )
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=4,
        help="Number of repeated fits used for stability summaries.",
    )
    parser.add_argument(
        "--stability-mcmc-iterations",
        type=int,
        default=5,
        help="Optional number of lightweight MCMC equilibration iterations after each fitted state.",
    )
    parser.add_argument(
        "--stability-multiflip",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use graph-tool multiflip MCMC when running stability sweeps if available.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Additional Hydra-style overrides applied to each dataset config.",
    )
    parser.add_argument(
        "--dataset-split-override",
        choices=("random", "fixed", "predefined"),
        default=None,
        help="Optional override for cfg.dataset.split inside this script.",
    )
    return parser.parse_args()


def resolve_dataset_request(dataset_request: str) -> DatasetRequest:
    raw_request = str(dataset_request).strip()
    if not raw_request:
        raise ValueError("Dataset request must be non-empty.")

    base_key = raw_request
    difficulty: str | None = None

    if ":" in raw_request:
        base_key, difficulty = raw_request.split(":", 1)
        base_key = base_key.strip()
        difficulty = difficulty.strip().lower()
    else:
        for ladder_key in LADDER_DIFFICULTIES:
            prefix = f"{ladder_key}_"
            if raw_request.startswith(prefix):
                base_key = ladder_key
                difficulty = raw_request[len(prefix) :].strip().lower()
                break

    if base_key in LADDER_DIFFICULTIES:
        resolved_difficulty = difficulty or "medium"
        valid_difficulties = LADDER_DIFFICULTIES[base_key]
        if resolved_difficulty not in valid_difficulties:
            raise ValueError(
                f"Unknown difficulty '{resolved_difficulty}' for {base_key}. "
                f"Choose from: {', '.join(valid_difficulties)}."
            )

        if base_key == "hsbm_ladder":
            dataset_name_override = f"hsbm_ladder_{resolved_difficulty}"
            dataset_root_override = f"${{oc.env:DATA_ROOT,./data}}/hsbm_{resolved_difficulty}"
        else:
            dataset_name_override = f"granularity_ladder_{resolved_difficulty}"
            dataset_root_override = f"${{oc.env:DATA_ROOT,./data}}/granularity_{resolved_difficulty}"

        return DatasetRequest(
            raw_request=raw_request,
            hydra_dataset_key=base_key,
            dataset_name_override=dataset_name_override,
            dataset_root_override=dataset_root_override,
            difficulty=resolved_difficulty,
        )

    if difficulty is not None:
        raise ValueError(
            f"Explicit difficulty was provided for unsupported dataset '{base_key}'. "
            "Only hsbm_ladder and granularity_ladder accept difficulty suffixes."
        )

    return DatasetRequest(
        raw_request=raw_request,
        hydra_dataset_key=base_key,
        dataset_name_override=None,
        dataset_root_override=None,
        difficulty=None,
    )


def dataset_request_overrides(dataset_request: DatasetRequest) -> list[str]:
    overrides: list[str] = []
    if dataset_request.dataset_name_override is not None:
        overrides.append(f"dataset.name={dataset_request.dataset_name_override}")
    if dataset_request.dataset_root_override is not None:
        overrides.append(f"dataset.root={dataset_request.dataset_root_override}")
    return overrides


def compose_cfg(dataset_request: DatasetRequest, force_reload: bool, extra_overrides: list[str]) -> DictConfig:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    overrides = [
        f"dataset={dataset_request.hydra_dataset_key}",
        "model=mp_gnn",
        "subgraph_policy=whole_graph",
        "experiment.self_loops=false",
        "dataset.multiprocess=false",
        "dataset.loader_params.num_workers=0",
        "dataset.loader_params.persistent_workers=false",
        "dataset.loader_params.prefetch_factor=null",
        *dataset_request_overrides(dataset_request),
        *extra_overrides,
    ]
    if force_reload is not None:
        overrides.append(f"dataset.force_reload={'true' if force_reload else 'false'}")

    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return hydra.compose(config_name="config", overrides=overrides)


def maybe_change_to_local_data_dir(cfg: DictConfig) -> None:
    is_local = torch.mps.is_available()
    if is_local and cfg.get("local_data_dir", None):
        os.chdir(cfg.local_data_dir)
        LOGGER.info("Changed working directory to local data dir: %s", os.getcwd())


def resolve_output_dir(output_dir: str) -> Path:
    output_path = Path(output_dir)
    if output_path.is_absolute():
        return output_path
    return PROJECT_ROOT / output_path


def collect_graph_records(loaders: list[tuple[str, Any]], max_graphs: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    combined_index = 0
    for split_name, loader_or_dataset in loaders:
        dataset = getattr(loader_or_dataset, "dataset", loader_or_dataset)
        for split_graph_index in range(len(dataset)):
            records.append(
                {
                    "split": split_name,
                    "split_graph_index": split_graph_index,
                    "combined_graph_index": combined_index,
                    "data": dataset[split_graph_index],
                }
            )
            combined_index += 1
            if max_graphs is not None and len(records) >= max_graphs:
                return records
    return records


def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(row.get("combined_graph_index", -1)))


def build_preprocessing_transforms(cfg: DictConfig) -> tuple[Any, Any, Any]:
    pre_transforms, pre_filters, transforms = fetch_transform(cfg)

    sg_transform = get_subgraph_transform(cfg)
    if sg_transform:
        pre_transforms = T.Compose([pre_transforms, sg_transform]) if pre_transforms else sg_transform

    if cfg.experiment.self_loops:
        replace_connectivity = ReplaceSelfLoops()
        pre_transforms = T.Compose([pre_transforms, replace_connectivity]) if pre_transforms else replace_connectivity

    return pre_transforms, pre_filters, transforms


def instantiate_predefined_split_datasets(cfg: DictConfig) -> list[tuple[str, Any]]:
    source = cfg.dataset.source
    pre_transforms, pre_filters, transforms = build_preprocessing_transforms(cfg)

    if source == "ZINC":
        return [
            (
                "train",
                ZINC(
                    root=cfg.dataset.root,
                    split="train",
                    subset=cfg.dataset.subset,
                    pre_transform=pre_transforms,
                    pre_filter=pre_filters,
                    transform=transforms,
                    force_reload=cfg.dataset.force_reload,
                ),
            ),
            (
                "val",
                ZINC(
                    root=cfg.dataset.root,
                    split="val",
                    subset=cfg.dataset.subset,
                    pre_transform=pre_transforms,
                    pre_filter=pre_filters,
                    transform=transforms,
                    force_reload=False,
                ),
            ),
            (
                "test",
                ZINC(
                    root=cfg.dataset.root,
                    split="test",
                    subset=cfg.dataset.subset,
                    pre_transform=pre_transforms,
                    pre_filter=pre_filters,
                    transform=transforms,
                    force_reload=False,
                ),
            ),
        ]

    raise NotImplementedError(
        f"Predefined split handling is not implemented in hierarchy_scoring.py for dataset source '{source}'."
    )


def build_dataset_records(cfg: DictConfig, max_graphs: int | None) -> list[dict[str, Any]]:
    split_mode = str(cfg.dataset.split)
    if split_mode == "predefined":
        split_datasets = instantiate_predefined_split_datasets(cfg)
        return collect_graph_records([(split_name, dataset) for split_name, dataset in split_datasets], max_graphs=max_graphs)

    train_loader, val_loader, test_loader, _, _ = get_data(cfg)
    return collect_graph_records(
        [("train", train_loader), ("val", val_loader), ("test", test_loader)],
        max_graphs=max_graphs,
    )


def canonicalize_edge_pairs(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    if edge_index is None or edge_index.numel() == 0:
        return []
    pairs = edge_index.t().detach().cpu().tolist()
    unique_pairs = {tuple(sorted((int(u), int(v)))) for u, v in pairs}
    return sorted(unique_pairs)


def extract_graph_id(data: Any, fallback_index: int) -> str:
    for attr in ("graph_id", "id", "name", "graph_name", "uid"):
        value = getattr(data, attr, None)
        if value is None:
            continue
        if torch.is_tensor(value):
            if value.numel() == 1:
                return str(value.view(-1)[0].item())
            return json.dumps(value.detach().cpu().view(-1).tolist())
        return str(value)
    return str(fallback_index)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def safe_state_entropy(state: Any, **kwargs: Any) -> float:
    try:
        return safe_float(state.entropy(**kwargs))
    except Exception:
        return float("nan")


def safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except Exception:
        return json.dumps(str(value))


def cfg_to_serializable(cfg: DictConfig) -> Any:
    """Convert a Hydra config to plain containers without resolving Hydra runtime interpolations."""
    return OmegaConf.to_container(cfg, resolve=False)


def auc_from_scores(pos_scores: list[float], neg_scores: list[float]) -> float:
    if not pos_scores or not neg_scores:
        return float("nan")
    wins = 0.0
    total = len(pos_scores) * len(neg_scores)
    for pos in pos_scores:
        for neg in neg_scores:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return float(wins / total) if total else float("nan")


def summarize_array(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    arr = np.asarray(values, dtype=float)
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
    }


def ensure_graph_tool_available() -> None:
    if GRAPH_TOOL_IMPORT_ERROR is not None:
        raise ImportError(
            "graph_tool is required for this script but is not available in the current Python environment."
        ) from GRAPH_TOOL_IMPORT_ERROR


def configure_graph_tool_threads(num_threads: int) -> None:
    try:
        if openmp_enabled():
            openmp_set_num_threads(max(1, int(num_threads)))
    except Exception:
        LOGGER.warning("Failed to configure graph-tool OpenMP threads", exc_info=True)


def pyg_to_graph_tool(data: Any, Graph: Any) -> tuple[Any, list[tuple[int, int]], int]:
    num_nodes = int(data.num_nodes)
    all_pairs = canonicalize_edge_pairs(data.edge_index)
    removed_self_loops = sum(1 for u, v in all_pairs if u == v)
    simple_pairs = [(u, v) for u, v in all_pairs if u != v]

    gt_graph = Graph(directed=False)
    gt_graph.add_vertex(num_nodes)
    if simple_pairs:
        gt_graph.add_edge_list(simple_pairs)
    return gt_graph, simple_pairs, removed_self_loops


def safe_num_blocks(state: Any) -> float:
    for method_name in ("get_nonempty_B", "get_B"):
        method = getattr(state, method_name, None)
        if callable(method):
            try:
                return safe_float(method())
            except Exception:
                continue
    return float("nan")


def extract_nested_level_block_counts(nested_state: Any) -> list[int]:
    try:
        memberships = nested_state.get_bs()
    except Exception:
        return []

    block_counts: list[int] = []
    for membership in memberships:
        try:
            arr = np.asarray(membership)
            block_counts.append(int(np.unique(arr).size))
        except Exception:
            block_counts.append(-1)
    return block_counts


def fit_sbm_states(gt_graph: Any, minimize_blockmodel_dl: Any, minimize_nested_blockmodel_dl: Any) -> tuple[Any, Any]:
    state_args = {"deg_corr": True}
    flat_state = minimize_blockmodel_dl(gt_graph, state_args=state_args)
    nested_state = minimize_nested_blockmodel_dl(gt_graph, state_args=state_args)
    return flat_state, nested_state


def collect_fit_metrics(flat_state: Any, nested_state: Any) -> dict[str, Any]:
    flat_entropy = safe_state_entropy(flat_state)
    nested_entropy = safe_state_entropy(nested_state)
    flat_likelihood_only = safe_state_entropy(flat_state, dl=False)
    nested_likelihood_only = safe_state_entropy(nested_state, dl=False)

    level_block_counts = extract_nested_level_block_counts(nested_state)
    level_entropies: list[float] = []
    for level_idx in range(len(level_block_counts)):
        try:
            level_entropies.append(safe_float(nested_state.level_entropy(level_idx)))
        except Exception:
            level_entropies.append(float("nan"))

    return {
        "flat_entropy": flat_entropy,
        "flat_likelihood_only": flat_likelihood_only,
        "flat_model_dl_only": flat_entropy - flat_likelihood_only,
        "flat_num_blocks": safe_num_blocks(flat_state),
        "flat_effective_num_blocks": safe_float(getattr(flat_state, "get_Be", lambda: float("nan"))()),
        "nested_entropy": nested_entropy,
        "nested_likelihood_only": nested_likelihood_only,
        "nested_model_dl_only": nested_entropy - nested_likelihood_only,
        "nested_hierarchy_depth_total_levels": len(level_block_counts),
        "nested_hierarchy_depth_nontrivial_levels": int(sum(count > 1 for count in level_block_counts)),
        "nested_level_block_counts_json": safe_json_dumps(level_block_counts),
        "nested_level_entropies_json": safe_json_dumps(level_entropies),
        "nested_base_num_blocks": safe_float(level_block_counts[0]) if level_block_counts else float("nan"),
        "nested_top_num_blocks": safe_float(level_block_counts[-1]) if level_block_counts else float("nan"),
    }


def sample_non_edges(
    num_nodes: int,
    existing_edges: set[tuple[int, int]],
    sample_size: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    if num_nodes < 2 or sample_size <= 0:
        return []

    max_possible = num_nodes * (num_nodes - 1) // 2
    available = max_possible - len(existing_edges)
    if available <= 0:
        return []

    target = min(sample_size, available)
    sampled: set[tuple[int, int]] = set()
    max_attempts = max(target * 50, 1000)
    attempts = 0
    while len(sampled) < target and attempts < max_attempts:
        u = rng.randrange(num_nodes)
        v = rng.randrange(num_nodes)
        if u == v:
            attempts += 1
            continue
        edge = (u, v) if u < v else (v, u)
        if edge in existing_edges or edge in sampled:
            attempts += 1
            continue
        sampled.add(edge)
        attempts += 1

    if len(sampled) < target:
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                edge = (u, v)
                if edge in existing_edges or edge in sampled:
                    continue
                sampled.add(edge)
                if len(sampled) >= target:
                    break
            if len(sampled) >= target:
                break
    return sorted(sampled)


def build_graph_from_edges(num_nodes: int, edges: list[tuple[int, int]], Graph: Any) -> Any:
    gt_graph = Graph(directed=False)
    gt_graph.add_vertex(num_nodes)
    if edges:
        gt_graph.add_edge_list(edges)
    return gt_graph


def compute_link_prediction_metrics(state: Any, positive_edges: list[tuple[int, int]], negative_edges: list[tuple[int, int]]) -> dict[str, Any]:
    pos_scores = [safe_float(state.get_edges_prob(missing=[edge], spurious=[])) for edge in positive_edges]
    neg_scores = [safe_float(state.get_edges_prob(missing=[edge], spurious=[])) for edge in negative_edges]

    metrics = {
        "pos_scores_json": safe_json_dumps(pos_scores),
        "neg_scores_json": safe_json_dumps(neg_scores),
        "joint_pos_logprob": safe_float(state.get_edges_prob(missing=positive_edges, spurious=[])),
        "joint_neg_logprob": safe_float(state.get_edges_prob(missing=negative_edges, spurious=[])),
        "score_gap_mean": safe_float(np.mean(pos_scores) - np.mean(neg_scores)) if pos_scores and neg_scores else float("nan"),
        "auc": auc_from_scores(pos_scores, neg_scores),
        "num_positive_edges": len(positive_edges),
        "num_negative_edges": len(negative_edges),
    }
    metrics.update(summarize_array("pos_score", pos_scores))
    metrics.update(summarize_array("neg_score", neg_scores))
    return metrics


def run_heldout_edge_prediction(
    *,
    num_nodes: int,
    full_edges: list[tuple[int, int]],
    Graph: Any,
    minimize_blockmodel_dl: Any,
    minimize_nested_blockmodel_dl: Any,
    rng: random.Random,
    heldout_edge_fraction: float,
    heldout_max_pairs: int,
) -> dict[str, Any]:
    edge_count = len(full_edges)
    if num_nodes < 2 or edge_count < 4:
        return {}

    holdout_count = max(1, int(round(edge_count * heldout_edge_fraction)))
    holdout_count = min(holdout_count, heldout_max_pairs, max(edge_count - 1, 1))
    if holdout_count <= 0:
        return {}

    shuffled_edges = list(full_edges)
    rng.shuffle(shuffled_edges)
    heldout_edges = sorted(shuffled_edges[:holdout_count])
    reduced_edges = sorted(shuffled_edges[holdout_count:])
    non_edges = sample_non_edges(num_nodes, set(full_edges), holdout_count, rng)
    if not reduced_edges or len(non_edges) < len(heldout_edges):
        return {}

    reduced_graph = build_graph_from_edges(num_nodes, reduced_edges, Graph)
    flat_state, nested_state = fit_sbm_states(
        reduced_graph,
        minimize_blockmodel_dl=minimize_blockmodel_dl,
        minimize_nested_blockmodel_dl=minimize_nested_blockmodel_dl,
    )

    flat_metrics = compute_link_prediction_metrics(flat_state, heldout_edges, non_edges)
    nested_metrics = compute_link_prediction_metrics(nested_state, heldout_edges, non_edges)

    results: dict[str, Any] = {
        "heldout_edge_fraction_actual": safe_float(len(heldout_edges) / max(edge_count, 1)),
        "heldout_positive_edges_json": safe_json_dumps(heldout_edges),
        "heldout_negative_edges_json": safe_json_dumps(non_edges),
        "heldout_reduced_num_edges": len(reduced_edges),
    }
    for prefix, metrics in (("flat_heldout", flat_metrics), ("nested_heldout", nested_metrics)):
        for key, value in metrics.items():
            results[f"{prefix}_{key}"] = value

    results["heldout_auc_delta_nested_minus_flat"] = (
        safe_float(results["nested_heldout_auc"] - results["flat_heldout_auc"])
        if not math.isnan(results["nested_heldout_auc"]) and not math.isnan(results["flat_heldout_auc"])
        else float("nan")
    )
    results["heldout_score_gap_delta_nested_minus_flat"] = (
        safe_float(results["nested_heldout_score_gap_mean"] - results["flat_heldout_score_gap_mean"])
        if not math.isnan(results["nested_heldout_score_gap_mean"]) and not math.isnan(results["flat_heldout_score_gap_mean"])
        else float("nan")
    )
    return results


def maybe_seed_graph_tool(seed_rng: Any, seed: int) -> None:
    try:
        seed_rng(int(seed))
    except Exception:
        LOGGER.debug("graph_tool.seed_rng(%s) failed", seed, exc_info=True)


def maybe_run_equilibration(state: Any, mcmc_equilibrate: Any, iterations: int, multiflip: bool) -> None:
    if iterations <= 0:
        return
    try:
        mcmc_equilibrate(
            state,
            force_niter=iterations,
            wait=max(1, iterations),
            nbreaks=1,
            multiflip=multiflip,
        )
    except Exception:
        LOGGER.debug("MCMC equilibration failed", exc_info=True)


def run_stability_checks(
    *,
    gt_graph: Any,
    repeats: int,
    seed: int,
    minimize_blockmodel_dl: Any,
    minimize_nested_blockmodel_dl: Any,
    mcmc_equilibrate: Any,
    seed_rng: Any,
    stability_mcmc_iterations: int,
    stability_multiflip: bool,
) -> dict[str, Any]:
    if repeats <= 1:
        return {}

    flat_entropies: list[float] = []
    flat_blocks: list[float] = []
    nested_entropies: list[float] = []
    nested_nontrivial_depths: list[float] = []
    nested_base_blocks: list[float] = []
    nested_level_signatures: list[list[int]] = []

    for repeat_idx in range(repeats):
        maybe_seed_graph_tool(seed_rng, seed + repeat_idx)
        flat_state, nested_state = fit_sbm_states(
            gt_graph,
            minimize_blockmodel_dl=minimize_blockmodel_dl,
            minimize_nested_blockmodel_dl=minimize_nested_blockmodel_dl,
        )
        maybe_run_equilibration(flat_state, mcmc_equilibrate, stability_mcmc_iterations, stability_multiflip)
        maybe_run_equilibration(nested_state, mcmc_equilibrate, stability_mcmc_iterations, stability_multiflip)

        flat_entropies.append(safe_state_entropy(flat_state))
        flat_blocks.append(safe_num_blocks(flat_state))
        nested_entropies.append(safe_state_entropy(nested_state))

        level_counts = extract_nested_level_block_counts(nested_state)
        nested_level_signatures.append(level_counts)
        nested_nontrivial_depths.append(float(sum(count > 1 for count in level_counts)))
        nested_base_blocks.append(float(level_counts[0]) if level_counts else float("nan"))

    metrics = {
        "stability_num_repeats": repeats,
        "stability_nested_level_signatures_json": safe_json_dumps(nested_level_signatures),
        "stability_num_unique_nested_level_signatures": len({tuple(sig) for sig in nested_level_signatures}),
    }
    metrics.update(summarize_array("stability_flat_entropy", flat_entropies))
    metrics.update(summarize_array("stability_flat_blocks", flat_blocks))
    metrics.update(summarize_array("stability_nested_entropy", nested_entropies))
    metrics.update(summarize_array("stability_nested_nontrivial_depth", nested_nontrivial_depths))
    metrics.update(summarize_array("stability_nested_base_blocks", nested_base_blocks))
    return metrics


def initialize_result_row(
    *,
    dataset_name: str,
    dataset_key: str,
    graph_record: dict[str, Any],
    graph_id: str,
    num_nodes: int,
    num_edges: int,
    removed_self_loops: int,
) -> dict[str, Any]:
    density = (2.0 * num_edges / (num_nodes * (num_nodes - 1))) if num_nodes > 1 else float("nan")
    return {
        "dataset_name": dataset_name,
        "dataset_key": dataset_key,
        "split": graph_record["split"],
        "split_graph_index": graph_record["split_graph_index"],
        "combined_graph_index": graph_record["combined_graph_index"],
        "graph_id": graph_id,
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "graph_density": density,
        "removed_self_loops_from_input": removed_self_loops,
        "status": "pending",
        "error_message": "",
    }


def serialize_graph_record(graph_record: dict[str, Any]) -> dict[str, Any]:
    data = graph_record["data"]
    graph_id = extract_graph_id(data, graph_record["combined_graph_index"])
    num_nodes = int(data.num_nodes)
    all_pairs = canonicalize_edge_pairs(data.edge_index)
    removed_self_loops = sum(1 for u, v in all_pairs if u == v)
    simple_edges = [(u, v) for u, v in all_pairs if u != v]
    return {
        "split": graph_record["split"],
        "split_graph_index": graph_record["split_graph_index"],
        "combined_graph_index": graph_record["combined_graph_index"],
        "graph_id": graph_id,
        "num_nodes": num_nodes,
        "num_edges": len(simple_edges),
        "removed_self_loops_from_input": removed_self_loops,
        "simple_edges": simple_edges,
    }


def compute_core_row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    flat_entropy = safe_float(row.get("flat_entropy"))
    nested_entropy = safe_float(row.get("nested_entropy"))
    num_edges = max(int(row.get("num_edges", 0)), 1)
    num_nodes = max(int(row.get("num_nodes", 0)), 1)
    delta = flat_entropy - nested_entropy
    return {
        "delta_entropy": delta,
        "delta_entropy_per_edge": delta / num_edges,
        "delta_entropy_per_node": delta / num_nodes,
    }


def finalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    default_nan_fields = {
        "heldout_edge_fraction_actual": float("nan"),
        "heldout_reduced_num_edges": float("nan"),
        "heldout_auc_delta_nested_minus_flat": float("nan"),
        "heldout_score_gap_delta_nested_minus_flat": float("nan"),
        "flat_heldout_auc": float("nan"),
        "nested_heldout_auc": float("nan"),
        "flat_heldout_score_gap_mean": float("nan"),
        "nested_heldout_score_gap_mean": float("nan"),
        "flat_heldout_joint_pos_logprob": float("nan"),
        "flat_heldout_joint_neg_logprob": float("nan"),
        "nested_heldout_joint_pos_logprob": float("nan"),
        "nested_heldout_joint_neg_logprob": float("nan"),
        "flat_heldout_num_positive_edges": float("nan"),
        "flat_heldout_num_negative_edges": float("nan"),
        "nested_heldout_num_positive_edges": float("nan"),
        "nested_heldout_num_negative_edges": float("nan"),
        "flat_heldout_pos_score_mean": float("nan"),
        "flat_heldout_pos_score_std": float("nan"),
        "flat_heldout_pos_score_min": float("nan"),
        "flat_heldout_pos_score_max": float("nan"),
        "flat_heldout_neg_score_mean": float("nan"),
        "flat_heldout_neg_score_std": float("nan"),
        "flat_heldout_neg_score_min": float("nan"),
        "flat_heldout_neg_score_max": float("nan"),
        "nested_heldout_pos_score_mean": float("nan"),
        "nested_heldout_pos_score_std": float("nan"),
        "nested_heldout_pos_score_min": float("nan"),
        "nested_heldout_pos_score_max": float("nan"),
        "nested_heldout_neg_score_mean": float("nan"),
        "nested_heldout_neg_score_std": float("nan"),
        "nested_heldout_neg_score_min": float("nan"),
        "nested_heldout_neg_score_max": float("nan"),
        "stability_num_repeats": float("nan"),
        "stability_num_unique_nested_level_signatures": float("nan"),
        "stability_flat_entropy_mean": float("nan"),
        "stability_flat_entropy_std": float("nan"),
        "stability_flat_entropy_min": float("nan"),
        "stability_flat_entropy_max": float("nan"),
        "stability_flat_blocks_mean": float("nan"),
        "stability_flat_blocks_std": float("nan"),
        "stability_flat_blocks_min": float("nan"),
        "stability_flat_blocks_max": float("nan"),
        "stability_nested_entropy_mean": float("nan"),
        "stability_nested_entropy_std": float("nan"),
        "stability_nested_entropy_min": float("nan"),
        "stability_nested_entropy_max": float("nan"),
        "stability_nested_nontrivial_depth_mean": float("nan"),
        "stability_nested_nontrivial_depth_std": float("nan"),
        "stability_nested_nontrivial_depth_min": float("nan"),
        "stability_nested_nontrivial_depth_max": float("nan"),
        "stability_nested_base_blocks_mean": float("nan"),
        "stability_nested_base_blocks_std": float("nan"),
        "stability_nested_base_blocks_min": float("nan"),
        "stability_nested_base_blocks_max": float("nan"),
    }
    for key, value in default_nan_fields.items():
        row.setdefault(key, value)
    row.setdefault("heldout_positive_edges_json", "")
    row.setdefault("heldout_negative_edges_json", "")
    row.setdefault("flat_heldout_pos_scores_json", "")
    row.setdefault("flat_heldout_neg_scores_json", "")
    row.setdefault("nested_heldout_pos_scores_json", "")
    row.setdefault("nested_heldout_neg_scores_json", "")
    row.setdefault("stability_nested_level_signatures_json", "")
    row.setdefault("error_traceback", "")
    row.setdefault("robustness_ran", False)
    return row


def write_rows_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sort_rows(rows)
    fieldnames = sorted({key for row in sorted_rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(row)


def save_histogram(values: list[float], title: str, xlabel: str, output_path: Path, bins: int) -> None:
    finite_values = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite_values:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.hist(finite_values, bins=bins, color="#3A6EA5", edgecolor="white", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_dataset_plots(rows: list[dict[str, Any]], output_dir: Path, bins: int) -> None:
    rows = sort_rows(rows)
    save_histogram(
        values=[row.get("delta_entropy", float("nan")) for row in rows],
        title="Flat minus Nested Description Length",
        xlabel="delta entropy",
        output_path=output_dir / "delta_entropy_hist.png",
        bins=bins,
    )


def build_worker_task(
    *,
    dataset_name: str,
    dataset_key: str,
    payload: dict[str, Any],
    args: argparse.Namespace,
    robustness_limit: int,
) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "dataset_key": dataset_key,
        "payload": payload,
        "args": {
            "seed": args.seed,
            "run_robustness": args.run_robustness,
            "robustness_limit": robustness_limit,
            "heldout_edge_fraction": args.heldout_edge_fraction,
            "heldout_max_pairs": args.heldout_max_pairs,
            "stability_runs": args.stability_runs,
            "stability_mcmc_iterations": args.stability_mcmc_iterations,
            "stability_multiflip": args.stability_multiflip,
            "graph_tool_threads": args.graph_tool_threads,
        },
    }


def score_graph_payload(task: dict[str, Any]) -> dict[str, Any]:
    ensure_graph_tool_available()

    dataset_name = task["dataset_name"]
    dataset_key = task["dataset_key"]
    payload = task["payload"]
    args = task["args"]

    configure_graph_tool_threads(args["graph_tool_threads"])

    row = initialize_result_row(
        dataset_name=dataset_name,
        dataset_key=dataset_key,
        graph_record=payload,
        graph_id=payload["graph_id"],
        num_nodes=payload["num_nodes"],
        num_edges=payload["num_edges"],
        removed_self_loops=payload["removed_self_loops_from_input"],
    )

    try:
        maybe_seed_graph_tool(seed_rng, args["seed"] + payload["combined_graph_index"])
        gt_graph = build_graph_from_edges(payload["num_nodes"], payload["simple_edges"], Graph)

        flat_state, nested_state = fit_sbm_states(
            gt_graph,
            minimize_blockmodel_dl=minimize_blockmodel_dl,
            minimize_nested_blockmodel_dl=minimize_nested_blockmodel_dl,
        )
        row.update(collect_fit_metrics(flat_state, nested_state))
        row.update(compute_core_row_metrics(row))
        row["status"] = "ok"

        should_run_robustness = (
            args["run_robustness"] and payload["combined_graph_index"] < args["robustness_limit"]
        )
        row["robustness_ran"] = bool(should_run_robustness)
        if should_run_robustness:
            random_seed = args["seed"] + 100_000 + payload["combined_graph_index"]
            rng = random.Random(random_seed)
            heldout_metrics = run_heldout_edge_prediction(
                num_nodes=payload["num_nodes"],
                full_edges=payload["simple_edges"],
                Graph=Graph,
                minimize_blockmodel_dl=minimize_blockmodel_dl,
                minimize_nested_blockmodel_dl=minimize_nested_blockmodel_dl,
                rng=rng,
                heldout_edge_fraction=args["heldout_edge_fraction"],
                heldout_max_pairs=args["heldout_max_pairs"],
            )
            stability_metrics = run_stability_checks(
                gt_graph=gt_graph,
                repeats=args["stability_runs"],
                seed=random_seed,
                minimize_blockmodel_dl=minimize_blockmodel_dl,
                minimize_nested_blockmodel_dl=minimize_nested_blockmodel_dl,
                mcmc_equilibrate=mcmc_equilibrate,
                seed_rng=seed_rng,
                stability_mcmc_iterations=args["stability_mcmc_iterations"],
                stability_multiflip=args["stability_multiflip"],
            )
            row.update(heldout_metrics)
            row.update(stability_metrics)
    except Exception as exc:
        row["status"] = "failed"
        row["error_message"] = f"{type(exc).__name__}: {exc}"
        row["error_traceback"] = traceback.format_exc(limit=6)

    return finalize_result_row(row)


def score_graphs_serial(
    *,
    dataset_name: str,
    dataset_key: str,
    payloads: list[dict[str, Any]],
    args: argparse.Namespace,
    csv_path: Path,
    robustness_limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress_bar = tqdm(payloads, desc=f"Scoring {dataset_name}", unit="graph")
    for payload in progress_bar:
        row = score_graph_payload(
            build_worker_task(
                dataset_name=dataset_name,
                dataset_key=dataset_key,
                payload=payload,
                args=args,
                robustness_limit=robustness_limit,
            )
        )
        if row["status"] == "ok":
            progress_bar.set_postfix_str(
                f"graph={payload['combined_graph_index']} delta={row['delta_entropy']:.3f}",
                refresh=False,
            )
        else:
            progress_bar.set_postfix_str(
                f"graph={payload['combined_graph_index']} failed",
                refresh=False,
            )
            LOGGER.error(
                "Failed processing dataset=%s graph_index=%s graph_id=%s error=%s",
                dataset_name,
                payload["combined_graph_index"],
                payload["graph_id"],
                row["error_message"],
            )
        rows.append(row)
        write_rows_csv(rows, csv_path)
    progress_bar.close()
    return rows


def score_graphs_parallel(
    *,
    dataset_name: str,
    dataset_key: str,
    payloads: list[dict[str, Any]],
    args: argparse.Namespace,
    csv_path: Path,
    robustness_limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress_bar = tqdm(total=len(payloads), desc=f"Scoring {dataset_name}", unit="graph")
    mp_context = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=args.num_workers, mp_context=mp_context) as executor:
        futures = {
            executor.submit(
                score_graph_payload,
                build_worker_task(
                    dataset_name=dataset_name,
                    dataset_key=dataset_key,
                    payload=payload,
                    args=args,
                    robustness_limit=robustness_limit,
                ),
            ): payload
            for payload in payloads
        }

        for future in as_completed(futures):
            payload = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = finalize_result_row(
                    {
                        "dataset_name": dataset_name,
                        "dataset_key": dataset_key,
                        "split": payload["split"],
                        "split_graph_index": payload["split_graph_index"],
                        "combined_graph_index": payload["combined_graph_index"],
                        "graph_id": payload["graph_id"],
                        "num_nodes": payload["num_nodes"],
                        "num_edges": payload["num_edges"],
                        "graph_density": (
                            (2.0 * payload["num_edges"] / (payload["num_nodes"] * (payload["num_nodes"] - 1)))
                            if payload["num_nodes"] > 1
                            else float("nan")
                        ),
                        "removed_self_loops_from_input": payload["removed_self_loops_from_input"],
                        "status": "failed",
                        "error_message": f"{type(exc).__name__}: {exc}",
                        "error_traceback": traceback.format_exc(limit=6),
                    }
                )

            if row["status"] == "ok":
                progress_bar.set_postfix_str(
                    f"graph={payload['combined_graph_index']} delta={row['delta_entropy']:.3f}",
                    refresh=False,
                )
            else:
                progress_bar.set_postfix_str(
                    f"graph={payload['combined_graph_index']} failed",
                    refresh=False,
                )
                LOGGER.error(
                    "Failed processing dataset=%s graph_index=%s graph_id=%s error=%s",
                    dataset_name,
                    payload["combined_graph_index"],
                    payload["graph_id"],
                    row["error_message"],
                )

            rows.append(row)
            write_rows_csv(rows, csv_path)
            progress_bar.update(1)

    progress_bar.close()
    return rows
    save_histogram(
        values=[row.get("delta_entropy_per_edge", float("nan")) for row in rows],
        title="Flat minus Nested Description Length per Edge",
        xlabel="delta entropy / edge",
        output_path=output_dir / "delta_entropy_per_edge_hist.png",
        bins=bins,
    )
    save_histogram(
        values=[row.get("delta_entropy_per_node", float("nan")) for row in rows],
        title="Flat minus Nested Description Length per Node",
        xlabel="delta entropy / node",
        output_path=output_dir / "delta_entropy_per_node_hist.png",
        bins=bins,
    )


def process_dataset(
    args: argparse.Namespace,
    dataset_key: str,
) -> None:
    dataset_request = resolve_dataset_request(dataset_key)
    cfg = compose_cfg(dataset_request, force_reload=args.force_reload, extra_overrides=args.overrides)
    if args.dataset_split_override is not None:
        cfg.dataset.split = args.dataset_split_override
    maybe_change_to_local_data_dir(cfg)
    LOGGER.info(
        "Composed config for dataset_request=%s hydra_dataset=%s difficulty=%s dataset_name=%s",
        dataset_request.raw_request,
        dataset_request.hydra_dataset_key,
        dataset_request.difficulty,
        cfg.dataset.name,
    )

    LOGGER.info(
        "Loading dataset=%s with dataset.split=%s dataset.multiprocess=%s loader.num_workers=%s persistent_workers=%s",
        dataset_request.raw_request,
        cfg.dataset.get("split", None),
        cfg.dataset.get("multiprocess", None),
        cfg.dataset.loader_params.get("num_workers", None),
        cfg.dataset.loader_params.get("persistent_workers", None),
    )
    graph_records = build_dataset_records(cfg, max_graphs=args.max_graphs)
    LOGGER.info("Materialized %s graph records for dataset=%s", len(graph_records), dataset_request.raw_request)
    configure_graph_tool_threads(args.graph_tool_threads)
    LOGGER.info("Configured graph-tool OpenMP threads=%s", max(1, int(args.graph_tool_threads)))
    if args.num_workers > 1 and args.graph_tool_threads > 1:
        LOGGER.warning(
            "Using num_workers=%s with graph_tool_threads=%s can oversubscribe CPUs heavily.",
            args.num_workers,
            args.graph_tool_threads,
        )

    dataset_name = str(cfg.dataset.name)
    dataset_output_root = resolve_output_dir(args.output_dir)
    dataset_output_dir = dataset_output_root / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = dataset_output_dir / "run_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_key": dataset_request.hydra_dataset_key,
                "dataset_request": dataset_request.raw_request,
                "dataset_difficulty": dataset_request.difficulty,
                "dataset_name": dataset_name,
                "num_graphs_requested": len(graph_records),
                "seed": args.seed,
                "run_robustness": args.run_robustness,
                "robustness_graphs": args.robustness_graphs,
                "hydra_config": cfg_to_serializable(cfg),
                "cli_args": vars(args),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    csv_path = dataset_output_dir / "graph_scores.csv"
    robustness_limit = min(args.robustness_graphs, len(graph_records)) if args.run_robustness else 0

    if graph_records:
        first_graph = graph_records[0]["data"]
        if hasattr(first_graph, "sg_nodes_orig"):
            LOGGER.warning(
                "Dataset=%s yielded SubgraphData-style cached graphs although whole-graph loading was requested. "
                "This indicates a stale processed cache. Re-run with --force-reload to rebuild plain graphs.",
                dataset_name,
            )

    payloads = [serialize_graph_record(graph_record) for graph_record in graph_records]
    LOGGER.info(
        "Prepared %s graph payloads for dataset=%s using num_workers=%s",
        len(payloads),
        dataset_name,
        args.num_workers,
    )

    if args.num_workers <= 1:
        rows = score_graphs_serial(
            dataset_name=dataset_name,
            dataset_key=dataset_request.hydra_dataset_key,
            payloads=payloads,
            args=args,
            csv_path=csv_path,
            robustness_limit=robustness_limit,
        )
    else:
        rows = score_graphs_parallel(
            dataset_name=dataset_name,
            dataset_key=dataset_request.hydra_dataset_key,
            payloads=payloads,
            args=args,
            csv_path=csv_path,
            robustness_limit=robustness_limit,
        )

    save_dataset_plots(rows, dataset_output_dir, bins=args.hist_bins)
    LOGGER.info(
        "Finished dataset=%s with %s graphs. Results: %s",
        dataset_name,
        len(rows),
        csv_path,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    ensure_graph_tool_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        LOGGER.debug("torch.set_num_interop_threads(1) failed", exc_info=True)

    for dataset_key in args.datasets:
        process_dataset(args=args, dataset_key=dataset_key)


if __name__ == "__main__":
    main()
