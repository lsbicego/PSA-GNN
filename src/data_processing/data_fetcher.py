
import torch
from .data_classes import (
    GraphNet,
    TUDiskDataset,
    MalNetTinyCustom,
)
from torch_geometric.datasets import TUDataset
from torch.utils.data import random_split
from .data_utils import fetch_transform, ReplaceSelfLoops, get_dataset_stats, compute_subgraph_statistics
from torch_geometric.loader import DataLoader

from torch_geometric import transforms as T
from src.data_processing.subgraph_transform import SubgraphTransform
import torch



from torch.serialization import add_safe_globals
from torch_geometric.data.data import Data, DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import BaseStorage, EdgeStorage, GlobalStorage, NodeStorage

add_safe_globals([
    Data,
    DataEdgeAttr,
    DataTensorAttr,
    BaseStorage,
    NodeStorage,
    EdgeStorage,
    GlobalStorage,
])


def get_subgraph_transform(cfg):
    """Build the subgraph preprocessing transform for the configured model.

    Args:
        cfg: Experiment configuration with model and subgraph-policy settings.

    Returns:
        A subgraph transform instance for subgraph-based models, or ``None`` for
        models that operate on the original graph directly.
    """
    if cfg.model.name in ["pool_gnn",
                            "ip_gnn",
                            "psa_gnn"]:
        sg_transform = SubgraphTransform(cfg)
        return sg_transform
    else:
        return None


def get_data(cfg):
    """Load the configured dataset, apply preprocessing, and create loaders.

    Args:
        cfg: Experiment configuration describing the dataset source, transforms,
            split strategy, and loader parameters.

    Returns:
        A tuple ``(train_loader, val_loader, test_loader, stats, sg_stats)``,
        where ``stats`` contains dataset metadata and ``sg_stats`` contains
        subgraph statistics for subgraph-based models or ``None`` otherwise.
    """
    NAME = cfg.dataset.name
    SOURCE = cfg.dataset.source

    pre_transforms, pre_filters, transforms = fetch_transform(cfg)
    train_data = None
    val_data = None
    test_data = None

    sg_transform = get_subgraph_transform(cfg)
    if sg_transform:
        pre_transforms = T.Compose([pre_transforms, sg_transform]) if pre_transforms else sg_transform
        print("After Subgraph Transforms: ", pre_transforms)

    if cfg.experiment.self_loops:
        replace_connectivity = ReplaceSelfLoops()
        pre_transforms = T.Compose([pre_transforms, replace_connectivity]) if pre_transforms else replace_connectivity
        print("After Self-loop Transform: ", pre_transforms)

    if SOURCE == 'TUDataset':
        dataset = TUDataset(root=cfg.dataset.root,
                            name=NAME,
                            pre_transform=pre_transforms,
                            pre_filter=pre_filters,
                            transform=transforms,
                            use_edge_attr=True,
                            force_reload=cfg.dataset.force_reload)
    elif SOURCE == 'TUDiskDataset':
        dataset = TUDiskDataset(name=NAME,
                               root=cfg.dataset.root,
                               pre_transform=pre_transforms,
                               pre_filter=pre_filters,
                               transform=transforms,
                               multiprocess=cfg.dataset.multiprocess,
                               force_reload=cfg.dataset.force_reload)
    elif SOURCE == "Malnet":
        dataset = MalNetTinyCustom(root=cfg.dataset.root,
                                   pre_transform=pre_transforms,
                                   pre_filter=pre_filters,
                                   transform=transforms,
                                   force_reload=cfg.dataset.force_reload,
                                   multiprocess=cfg.dataset.multiprocess)
    elif SOURCE == 'Own':
        dataset = GraphNet(root=cfg.dataset.root,
                           pre_transform=pre_transforms,
                           pre_filter=pre_filters,
                           subsample_graphs=cfg.dataset.get('subsample_graphs', None),
                           transform=transforms,
                           multiprocess=cfg.dataset.multiprocess,
                           force_reload=cfg.dataset.force_reload)
    elif SOURCE == "ZINC":
        from torch_geometric.datasets import ZINC
        train_data = ZINC(root=cfg.dataset.root,
                       split="train",
                       subset=cfg.dataset.subset,
                       pre_transform=pre_transforms,
                       pre_filter=pre_filters,
                       transform=transforms,
                       force_reload=cfg.dataset.force_reload)
        val_data = ZINC(root=cfg.dataset.root,
                     split="val",
                     subset=cfg.dataset.subset,
                     pre_transform=pre_transforms,
                     pre_filter=pre_filters,
                     transform=transforms,
                     force_reload=False, # because train already transformed all splits
                    )
        test_data = ZINC(root=cfg.dataset.root,
                      split="test",
                      subset=cfg.dataset.subset,
                      pre_transform=pre_transforms,
                      pre_filter=pre_filters,
                      transform=transforms,
                      force_reload=False, # because train already transformed all splits
                      )
        dataset = train_data + val_data + test_data
    else:
        raise ValueError(f"Unsupported dataset source: {SOURCE} for dataset {NAME}")


    print(f"Dataset: {NAME}, Number of graphs: {len(dataset)}")

    example = dataset[0]

    print("Example graph:")
    print(example)

    stats = get_dataset_stats(example, cfg)
    stats['num_graphs'] = len(dataset)
    stats["max_nodes"] = max([data.num_nodes for data in dataset])

    # calculate subgraph statistics if its a subgraph model
    if sg_transform is not None:
        sg_stats = compute_subgraph_statistics(dataset)
        stats.update(sg_stats)
        print("Subgraph statistics: ", sg_stats)


    # split dataset
    if train_data is None or val_data is None or test_data is None:
        if cfg.dataset.split == "random":
            train_data, val_data, test_data = random_split(
                dataset,
                lengths=[0.8, 0.1, 0.1],
                generator=torch.Generator().manual_seed(cfg.experiment.seed)
                            )
        elif cfg.dataset.split == "fixed":
            # just split first 80% in train, next 10% val, last 10% test
            n = len(dataset)
            train_data = dataset[:int(0.8 * n)]
            val_data = dataset[int(0.8 * n):int(0.9 * n)]
            test_data = dataset[int(0.9 * n):]
        elif cfg.dataset.split == "predefined":
            raise NotImplementedError(
                f"Dataset source {SOURCE} does not expose predefined splits for dataset {NAME}."
            )
        else:
            raise NotImplementedError(
                f"Unknown split {cfg.dataset.split}. Choose among fixed, random, and predefined."
            )
    print("Length after Splitting:")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # create data loaders
    train_loader = DataLoader(train_data,
                              batch_size=cfg.dataset.optimization_params.batch_size,
                              shuffle=True,
                              pin_memory=cfg.dataset.loader_params.get('pin_memory', False),
                              num_workers=cfg.dataset.loader_params.get('num_workers', 0),
                              persistent_workers=cfg.dataset.loader_params.get('persistent_workers', False),
                              prefetch_factor=cfg.dataset.loader_params.get('prefetch_factor', 2))
    val_loader = DataLoader(val_data,
                            batch_size=cfg.dataset.optimization_params.batch_size,
                            shuffle=False,
                            pin_memory=cfg.dataset.loader_params.get('pin_memory', False),
                            num_workers=cfg.dataset.loader_params.get('num_workers', 0),
                            persistent_workers=cfg.dataset.loader_params.get('persistent_workers', False),
                            prefetch_factor=cfg.dataset.loader_params.get('prefetch_factor', 2))
    test_loader = DataLoader(test_data,
                             batch_size=cfg.dataset.optimization_params.batch_size,
                             shuffle=False,
                             pin_memory=cfg.dataset.loader_params.get('pin_memory', False),
                             num_workers=cfg.dataset.loader_params.get('num_workers', 0),
                             persistent_workers=cfg.dataset.loader_params.get('persistent_workers', False),
                             prefetch_factor=cfg.dataset.loader_params.get('prefetch_factor', 2))
    return train_loader, val_loader, test_loader, stats, sg_stats if sg_transform is not None else None
