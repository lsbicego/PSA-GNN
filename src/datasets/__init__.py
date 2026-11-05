"""Synthetic hierarchical benchmarks using the original per-graph format."""

from .synthetic import (
    SyntheticHierarchicalData,
    generate_synthetic_dataset,
    pack_synthetic_dataset,
    save_synthetic_dataset,
)

__all__ = ["SyntheticHierarchicalData", "generate_synthetic_dataset",
           "save_synthetic_dataset", "pack_synthetic_dataset"]
