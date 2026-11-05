from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation.synthetic_datasets import (
    generate_degree_matched_granularity_dataset_v2,
    generate_hsbm_macro_topology_variable_dataset,
    save_graph_dataset_to_disk,
)


def generate_hsbm_ladder(output_root: Path, num_graphs: int, seed: int) -> None:
    graphs, metas = generate_hsbm_macro_topology_variable_dataset(
        num_graphs=num_graphs,
        seed=seed,
    )
    save_graph_dataset_to_disk(
        graphs=graphs,
        metas=metas,
        directory=str(output_root / "hsbm_final" / "raw"),
    )


def generate_granularity_ladder(output_root: Path, num_graphs: int, seed: int) -> None:
    graphs, metas = generate_degree_matched_granularity_dataset_v2(
        num_graphs=num_graphs,
        seed=seed,
    )
    save_graph_dataset_to_disk(
        graphs=graphs,
        metas=metas,
        directory=str(output_root / "granularity" / "raw"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the supported synthetic datasets.")
    parser.add_argument(
        "--dataset",
        choices=["hsbm_ladder", "granularity_ladder", "both"],
        default="both",
    )
    parser.add_argument("--output-root", default="data")
    parser.add_argument("--num-graphs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    if args.dataset in {"hsbm_ladder", "both"}:
        generate_hsbm_ladder(output_root=output_root, num_graphs=args.num_graphs, seed=args.seed)
    if args.dataset in {"granularity_ladder", "both"}:
        generate_granularity_ladder(output_root=output_root, num_graphs=args.num_graphs, seed=args.seed)


if __name__ == "__main__":
    main()
