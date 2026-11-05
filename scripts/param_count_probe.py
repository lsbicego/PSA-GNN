import argparse
import copy
import pathlib
import sys

import hydra
from omegaconf import OmegaConf
from torch_geometric import transforms as T
from torch_geometric.datasets import TUDataset, ZINC

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_processing.data_classes import (  # noqa: E402
    GraphNet,
    MalNetTinyCustom,
    TUDiskDataset,
)
from src.data_processing.data_fetcher import get_subgraph_transform  # noqa: E402
from src.data_processing.data_utils import (  # noqa: E402
    ReplaceSelfLoops,
    fetch_transform,
    get_dataset_stats,
)
from src.models.model_factory import get_model  # noqa: E402
from src.utils import count_parameters  # noqa: E402


HARDCODED_STATS_PROFILES = {}


def _build_dataset(cfg):
    source = cfg.dataset.source
    pre_transforms, pre_filters, transforms = fetch_transform(cfg)

    sg_transform = get_subgraph_transform(cfg)
    if sg_transform:
        pre_transforms = T.Compose([pre_transforms, sg_transform]) if pre_transforms else sg_transform

    if cfg.experiment.self_loops:
        replace_connectivity = ReplaceSelfLoops()
        pre_transforms = T.Compose([pre_transforms, replace_connectivity]) if pre_transforms else replace_connectivity

    if source == "TUDataset":
        return TUDataset(
            root=cfg.dataset.root,
            name=cfg.dataset.name,
            pre_transform=pre_transforms,
            pre_filter=pre_filters,
            transform=transforms,
            use_edge_attr=True,
            force_reload=cfg.dataset.force_reload,
        )
    if source == "TUDiskDataset":
        return TUDiskDataset(
            name=cfg.dataset.name,
            root=cfg.dataset.root,
            pre_transform=pre_transforms,
            pre_filter=pre_filters,
            transform=transforms,
            multiprocess=cfg.dataset.multiprocess,
            force_reload=cfg.dataset.force_reload,
        )
    if source == "Malnet":
        return MalNetTinyCustom(
            root=cfg.dataset.root,
            pre_transform=pre_transforms,
            pre_filter=pre_filters,
            transform=transforms,
            force_reload=cfg.dataset.force_reload,
            multiprocess=cfg.dataset.multiprocess,
        )
    if source == "Own":
        return GraphNet(
            root=cfg.dataset.root,
            pre_transform=pre_transforms,
            pre_filter=pre_filters,
            subsample_graphs=cfg.dataset.get("subsample_graphs", None),
            transform=transforms,
            multiprocess=cfg.dataset.multiprocess,
            force_reload=cfg.dataset.force_reload,
        )
    if source == "ZINC":
        return ZINC(
            root=cfg.dataset.root,
            split="train",
            subset=cfg.dataset.subset,
            pre_transform=pre_transforms,
            pre_filter=pre_filters,
            transform=transforms,
            force_reload=cfg.dataset.force_reload,
        )
    raise ValueError(f"Unsupported dataset source: {source}")


def _count_with_hidden_dim(cfg, stats, hidden_dim):
    cfg_local = copy.deepcopy(cfg)
    cfg_local.dataset.model_params.hidden_dim = hidden_dim
    model = get_model(cfg_local, stats)
    return count_parameters(model)


def _resolve_hardcoded_stats_profile(args, cfg):
    if args.stats_profile is not None:
        profile = HARDCODED_STATS_PROFILES.get(args.stats_profile)
        if profile is None:
            available = ", ".join(sorted(HARDCODED_STATS_PROFILES))
            raise ValueError(
                f"Unknown stats profile '{args.stats_profile}'. Available profiles: {available}"
            )
        return copy.deepcopy(profile)

    dataset_name = str(cfg.dataset.name).lower()
    if dataset_name in HARDCODED_STATS_PROFILES:
        return copy.deepcopy(HARDCODED_STATS_PROFILES[dataset_name])

    return None


def main():
    parser = argparse.ArgumentParser(description="Probe model parameter counts for Hydra configs.")
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--search-min", type=int, default=1)
    parser.add_argument("--search-max", type=int, default=512)
    parser.add_argument("--hidden-dims", type=int, nargs="*", default=None)
    parser.add_argument("--num-node-features", type=int, default=None)
    parser.add_argument("--num-edge-features", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--num-node-marks", type=int, default=0)
    parser.add_argument("--num-edge-marks", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument("--stats-profile", default=None)
    parser.add_argument("--list-stats-profiles", action="store_true")
    parser.add_argument("overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.list_stats_profiles:
        print("Available stats profiles:")
        for profile_name, stats in sorted(HARDCODED_STATS_PROFILES.items()):
            print(f"  {profile_name}: {stats}")
        return

    overrides = [arg for arg in args.overrides if arg != "--"]
    with hydra.initialize_config_dir(version_base=None, config_dir=str(ROOT / "conf")):
        cfg = hydra.compose(config_name=args.config_name, overrides=overrides)

    if (
        args.num_node_features is not None
        and args.num_edge_features is not None
        and args.num_classes is not None
    ):
        stats = {
            "num_node_features": args.num_node_features,
            "num_edge_features": args.num_edge_features,
            "num_classes": args.num_classes,
            "num_node_marks": args.num_node_marks,
            "num_edge_marks": args.num_edge_marks,
        }
        if args.max_nodes is not None:
            stats["max_nodes"] = args.max_nodes
    else:
        stats = _resolve_hardcoded_stats_profile(args, cfg)
        if stats is None:
            dataset = _build_dataset(cfg)
            example = dataset[0]
            stats = get_dataset_stats(example, cfg)
            stats["max_nodes"] = max(data.num_nodes for data in dataset)

    print("Resolved config:")
    print(OmegaConf.to_yaml(cfg))
    print("Stats:", stats)

    if args.hidden_dims:
        for hidden_dim in args.hidden_dims:
            params = _count_with_hidden_dim(cfg, stats, hidden_dim)
            print(f"hidden_dim={hidden_dim} params={params}")

    if args.budget is not None:
        lo = args.search_min
        hi = args.search_max
        best_hidden = None
        best_params = None

        while lo <= hi:
            mid = (lo + hi) // 2
            params = _count_with_hidden_dim(cfg, stats, mid)
            print(f"probe hidden_dim={mid} params={params}")
            if params <= args.budget:
                best_hidden = mid
                best_params = params
                lo = mid + 1
            else:
                hi = mid - 1

        print(f"best_hidden_dim={best_hidden} best_params={best_params} budget={args.budget}")


if __name__ == "__main__":
    main()
