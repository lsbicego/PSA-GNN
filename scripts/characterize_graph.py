"""Characterize a plain-text edge list without loading pickled graph objects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge-list", type=Path, required=True,
                        help="Text file with one integer 'u v' pair per line; # comments allowed.")
    parser.add_argument("--num-nodes", type=int, required=True,
                        help="Total number of vertices, including isolated vertices (IDs 0 to N-1).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-threads", type=int, default=1)
    args = parser.parse_args()
    try:
        edges = []
        for line_number, line in enumerate(args.edge_list.read_text().splitlines(), 1):
            fields = line.partition("#")[0].split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"Line {line_number}: expected two integer node IDs.")
            try:
                edges.append(tuple(map(int, fields)))
            except ValueError as exc:
                raise ValueError(f"Line {line_number}: node IDs must be integers.") from exc
        import torch
        from torch_geometric.data import Data
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.hierarchy import compute_hierarchy_stats
        edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).t().contiguous()
        data = Data(edge_index=edge_index, num_nodes=args.num_nodes)
        result = compute_hierarchy_stats(data, seed=args.seed, num_threads=args.num_threads)
    except (OSError, ImportError, ValueError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
