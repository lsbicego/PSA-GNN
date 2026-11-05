from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import hydra
import networkx as nx
import numpy as np
import os
import torch
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.data_fetcher import get_data

CONF_DIR = PROJECT_ROOT / "conf"


def _default_analysis_overrides(dataset_key: str) -> list[str]:
    return []


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _to_networkx_graph(data: Any) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    if data.edge_index is not None and data.edge_index.numel() > 0:
        graph.add_edges_from((int(u), int(v)) for u, v in data.edge_index.t().detach().cpu().tolist())
    return graph


def _label_from_data(data: Any) -> Any:
    y = getattr(data, "y", None)
    if y is None:
        return None
    y = y.detach().cpu()
    if y.numel() == 1:
        return y.view(-1)[0].item()
    return y.view(-1).tolist()


def _collect_graphs(dataset: Any) -> list[Any]:
    return [dataset[i] for i in range(len(dataset))]


def _dataset_task_type(task_name: str) -> str:
    task_lower = task_name.lower()
    if "multi_label" in task_lower or "multilabel" in task_lower:
        return "multilabel"
    if "classification" in task_lower:
        return "classification"
    if "multi" in task_lower and "regress" in task_lower:
        return "multi_target_regression"
    return "regression"


def _compose_base_cfg(
    dataset_key: str,
    force_reload: bool | None,
    extra_overrides: list[str] | None = None,
) -> DictConfig:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    overrides = [
        f"dataset={dataset_key}",
        "model=mp_gnn",
        "subgraph_policy=whole_graph",
        "experiment.self_loops=false",
    ]
    overrides.extend(_default_analysis_overrides(dataset_key))
    if force_reload is not None:
        overrides.append(f"dataset.force_reload={'true' if force_reload else 'false'}")
    if extra_overrides:
        overrides.extend(extra_overrides)

    with hydra.initialize_config_dir(version_base=None, config_dir=str(CONF_DIR)):
        return hydra.compose(config_name="config", overrides=overrides)


def _compute_dataset_analysis(
    graphs: list[Any],
    dataset_cfg: DictConfig,
    compute_expensive_metrics: bool,
    max_graphs_for_expensive_metrics: int | None,
    max_nodes_for_expensive_metrics: int | None,
) -> dict[str, Any]:
    task_type = _dataset_task_type(dataset_cfg.task)
    node_counts: list[float] = []
    edge_counts: list[float] = []
    mean_degree_per_graph: list[float] = []
    degree_values: list[float] = []
    density_values: list[float] = []
    component_counts: list[float] = []
    expensive_diameters: list[float] = []
    expensive_clusterings: list[float] = []
    expensive_transitivities: list[float] = []
    expensive_done = 0

    class_counts: Counter[int] = Counter()
    multilabel_positive_counts: np.ndarray | None = None
    multilabel_cardinality: list[float] = []
    regression_targets: list[np.ndarray] = []

    for graph_data in graphs:
        graph = _to_networkx_graph(graph_data)
        n = float(graph.number_of_nodes())
        m = float(graph.number_of_edges())
        node_counts.append(n)
        edge_counts.append(m)
        density_values.append(float(nx.density(graph)) if n > 1 else 0.0)

        degrees = [deg for _, deg in graph.degree()]
        mean_degree_per_graph.append(float(np.mean(degrees)) if degrees else 0.0)
        degree_values.extend(float(d) for d in degrees)
        component_counts.append(float(nx.number_connected_components(graph)) if n > 0 else 0.0)

        label = _label_from_data(graph_data)
        if task_type == "classification" and label is not None:
            class_counts[int(label)] += 1
        elif task_type == "multilabel" and label is not None:
            label_arr = np.asarray(label, dtype=float)
            if multilabel_positive_counts is None:
                multilabel_positive_counts = np.zeros_like(label_arr)
            multilabel_positive_counts += label_arr
            multilabel_cardinality.append(float(label_arr.sum()))
        elif task_type in {"regression", "multi_target_regression"} and label is not None:
            regression_targets.append(np.asarray(label, dtype=float))

        if (
            compute_expensive_metrics
            and (max_graphs_for_expensive_metrics is None or expensive_done < max_graphs_for_expensive_metrics)
            and (max_nodes_for_expensive_metrics is None or int(n) <= max_nodes_for_expensive_metrics)
            and int(n) > 0
        ):
            largest_cc = graph if nx.is_connected(graph) else graph.subgraph(max(nx.connected_components(graph), key=len)).copy()
            expensive_diameters.append(float(nx.diameter(largest_cc)) if largest_cc.number_of_nodes() > 1 else 0.0)
            expensive_clusterings.append(float(nx.average_clustering(graph)))
            expensive_transitivities.append(float(nx.transitivity(graph)))
            expensive_done += 1

    result: dict[str, Any] = {
        "dataset_name": dataset_cfg.name,
        "dataset_source": dataset_cfg.source,
        "task": dataset_cfg.task,
        "task_type_inferred": task_type,
        "num_graphs": len(graphs),
        "num_node_features": int(graphs[0].num_node_features) if graphs else 0,
        "num_edge_features": int(graphs[0].num_edge_features) if graphs else 0,
        "num_target_classes_cfg": int(dataset_cfg.get("num_classes", 1)),
        "nodes_per_graph": _summary(node_counts),
        "edges_per_graph": _summary(edge_counts),
        "mean_node_degree_per_graph": _summary(mean_degree_per_graph),
        "node_degree_overall": _summary(degree_values),
        "graph_density": _summary(density_values),
        "connected_components_per_graph": _summary(component_counts),
        "connected_graph_ratio": float(np.mean([c == 1 for c in component_counts])) if component_counts else None,
        "diameter_lcc": _summary(expensive_diameters),
        "average_clustering": _summary(expensive_clusterings),
        "transitivity": _summary(expensive_transitivities),
        "num_graphs_used_for_expensive_metrics": expensive_done,
        "metrics_scope": (
            "all_graphs"
            if (
                compute_expensive_metrics
                and max_graphs_for_expensive_metrics is None
                and max_nodes_for_expensive_metrics is None
            )
            else "capped_or_filtered"
        ),
    }

    if task_type == "classification":
        class_hist = {str(k): int(v) for k, v in sorted(class_counts.items())}
        total = max(len(graphs), 1)
        result["class_counts"] = class_hist
        result["class_ratios"] = {k: float(v / total) for k, v in class_hist.items()}
        result["largest_class_ratio"] = float(max(class_hist.values()) / total) if class_hist else None
    elif task_type == "multilabel":
        if multilabel_positive_counts is not None:
            total = max(len(graphs), 1)
            result["positive_counts_per_label"] = [int(v) for v in multilabel_positive_counts.tolist()]
            result["positive_ratio_per_label"] = [float(v / total) for v in multilabel_positive_counts.tolist()]
            result["label_cardinality_per_graph"] = _summary(multilabel_cardinality)
        else:
            result["positive_counts_per_label"] = None
            result["positive_ratio_per_label"] = None
            result["label_cardinality_per_graph"] = _summary([])
    else:
        if regression_targets:
            targets = np.vstack([t.reshape(1, -1) if t.ndim == 1 else t.reshape(1, -1) for t in regression_targets])
            result["target_mean"] = [float(v) for v in targets.mean(axis=0).tolist()]
            result["target_std"] = [float(v) for v in targets.std(axis=0).tolist()]
            result["target_min"] = [float(v) for v in targets.min(axis=0).tolist()]
            result["target_max"] = [float(v) for v in targets.max(axis=0).tolist()]
        else:
            result["target_mean"] = None
            result["target_std"] = None
            result["target_min"] = None
            result["target_max"] = None

    return result


def _print_dataset_result(result: dict[str, Any]) -> None:
    print(f"\n=== {result['dataset_name']} ({result['dataset_source']}) ===")
    print(f"Task: {result['task']} | Graphs: {result['num_graphs']}")
    print(
        f"Node feat dim: {result['num_node_features']} | "
        f"Edge feat dim: {result['num_edge_features']} | "
        f"Target classes (cfg): {result['num_target_classes_cfg']}"
    )
    print(
        f"Nodes/graph mean+-std: {result['nodes_per_graph']['mean']:.3f} +- {result['nodes_per_graph']['std']:.3f} | "
        f"Edges/graph mean+-std: {result['edges_per_graph']['mean']:.3f} +- {result['edges_per_graph']['std']:.3f}"
    )
    print(
        f"Degree mean+-std: {result['node_degree_overall']['mean']:.3f} +- {result['node_degree_overall']['std']:.3f} | "
        f"Connected ratio: {result['connected_graph_ratio']:.3f}"
    )
    if result["num_graphs_used_for_expensive_metrics"] > 0:
        print(
            f"Diameter(LCC) mean+-std: {result['diameter_lcc']['mean']:.3f} +- {result['diameter_lcc']['std']:.3f} | "
            f"Clustering mean+-std: {result['average_clustering']['mean']:.3f} +- {result['average_clustering']['std']:.3f}"
        )
    if "class_counts" in result:
        print(f"Class counts: {result['class_counts']}")
        print(f"Class ratios: {result['class_ratios']}")
    if "positive_counts_per_label" in result and result["positive_counts_per_label"] is not None:
        print(f"Positive counts per label: {result['positive_counts_per_label']}")
    if "target_mean" in result and result["target_mean"] is not None:
        print(f"Target mean: {result['target_mean']}")
        print(f"Target std: {result['target_std']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze datasets via get_data() and save summary JSON.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset config names from conf/dataset, e.g. reddit collab nci1",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="OUT/dataset_stats/dataset_summary.json",
        help="Output path for analysis JSON file.",
    )
    parser.add_argument(
        "--compute-expensive-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute diameter/clustering/transitivity on sampled graphs.",
    )
    parser.add_argument(
        "--max-graphs-for-expensive-metrics",
        type=int,
        default=None,
        help="Optional cap on graphs for expensive metrics. Default: no cap (all graphs).",
    )
    parser.add_argument(
        "--max-nodes-for-expensive-metrics",
        type=int,
        default=None,
        help="Optional node cap for expensive metrics. Default: no cap.",
    )
    parser.add_argument(
        "--force-reload",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Override dataset.force_reload.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=None,
        help="Additional Hydra-style overrides, e.g. dataset.root=/path/to/data/hsbm_final",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    all_results: dict[str, Any] = {"datasets": {}}

    for dataset_key in args.datasets:
        cfg = _compose_base_cfg(
            dataset_key=dataset_key,
            force_reload=args.force_reload,
            extra_overrides=args.overrides,
        )
        is_local = torch.mps.is_available()
        if is_local and cfg.get("local_data_dir", None) is not None:
            os.chdir(cfg.local_data_dir)
            print(f"Changed working directory to local data dir: {os.getcwd()}")
        print(f"\nLoading dataset '{dataset_key}' with model={cfg.model.name}, subgraph_policy={cfg.subgraph_policy.method}")
        train_loader, val_loader, test_loader, _, _ = get_data(cfg)
        split_sizes = {
            "train": len(train_loader.dataset),
            "val": len(val_loader.dataset),
            "test": len(test_loader.dataset),
        }
        graphs = _collect_graphs(train_loader.dataset) + _collect_graphs(val_loader.dataset) + _collect_graphs(test_loader.dataset)

        result = _compute_dataset_analysis(
            graphs=graphs,
            dataset_cfg=cfg.dataset,
            compute_expensive_metrics=args.compute_expensive_metrics,
            max_graphs_for_expensive_metrics=args.max_graphs_for_expensive_metrics,
            max_nodes_for_expensive_metrics=args.max_nodes_for_expensive_metrics,
        )
        result["split_sizes"] = split_sizes
        _print_dataset_result(result)
        all_results["datasets"][dataset_key] = result

    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)
    print(f"\nSaved dataset analysis JSON to: {output_path}")


if __name__ == "__main__":
    main()
