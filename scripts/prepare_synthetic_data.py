"""Generate original per-graph datasets or pack existing files for publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets import generate_synthetic_dataset, pack_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Call the original generator and saver.")
    pack = commands.add_parser("pack", help="Archive existing graph_N.pt files unchanged.")
    for command in (generate, pack):
        command.add_argument("--dataset", choices=("macro", "granular"), required=True)
        command.add_argument("--output", required=True, help="New or empty output directory.")
    generate.add_argument("--num-graphs", type=int, default=100)
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--parameter", action="append", default=[], metavar="NAME=JSON")
    pack.add_argument("--source", required=True, help="Dataset root or raw directory.")
    pack.add_argument("--generation-json", type=Path)
    pack.add_argument("--splits-json", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            parameters = {}
            for item in args.parameter:
                key, separator, value = item.partition("=")
                if not separator or not key or key in parameters or key in {"seed", "num_graphs"}:
                    raise ValueError(f"Invalid or duplicate --parameter {item!r}; use NAME=JSON.")
                parameters[key] = json.loads(value)
            dataset = generate_synthetic_dataset(args.output, name=args.dataset,
                                                 num_graphs=args.num_graphs, seed=args.seed, **parameters)
            print(f"Generated {len(dataset)} {dataset.name} graphs at {Path(args.output).resolve()}")
        else:
            generation = json.loads(args.generation_json.read_text()) if args.generation_json else None
            splits = json.loads(args.splits_json.read_text()) if args.splits_json else None
            output = pack_synthetic_dataset(args.source, args.output, name=args.dataset,
                                            generation=generation, split_indices=splits)
            print(f"Prepared {output / 'graphs.zip'} and manifest.json")
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
