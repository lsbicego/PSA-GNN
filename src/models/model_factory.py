from .mp_gnns import MP_GNN
from .subgraph_gnn import SubgraphGNN
from .hierarchical_gnns import HierarchicalDensePoolingGNN, HierarchicalSparsePoolingGNN
from .flat_pool_gnns import SubgraphPoolingGNN

def get_model(cfg, stats):
    """Build the model selected by the experiment configuration.

    Args:
        cfg: Experiment configuration containing the model name and
            hyperparameters.
        stats: Dataset metadata such as feature dimensions and output size.

    Returns:
        An initialized model instance matching ``cfg.model.name``.
    """
    if cfg.model.name == "mp_gnn":
        model = MP_GNN(cfg, stats)
    elif cfg.model.name == "pool_gnn":
        assert not cfg.model.within_cluster_mp, "For pool_gnn, cfg.model.within_cluster_mp must be false."
        assert cfg.model.number_within_mp == 0, "For pool_gnn, cfg.model.number_within_mp must be 0."
        model = SubgraphPoolingGNN(cfg, stats)
    elif cfg.model.name == "ip_gnn":
        assert cfg.model.within_cluster_mp, "For ip_gnn, cfg.model.within_cluster_mp must be true."
        assert cfg.model.number_within_mp > 0, "For ip_gnn, cfg.model.number_within_mp must be positive."
        model = SubgraphPoolingGNN(cfg, stats)
    elif cfg.model.name == "psa_gnn":
        assert cfg.model.mode == "psa", "For psa_gnn, cfg.model.mode must be set to 'psa'."
        model = SubgraphGNN(cfg, stats)
    elif cfg.model.name == "hierarchical_pool_gnn":
        if cfg.model.pooling_type.lower() in ['diffpool', 'mincut']:
            model = HierarchicalDensePoolingGNN(cfg, stats)
        elif cfg.model.pooling_type.lower() in ['topk', 'sag', "asap", "edge_pool"]:
            model = HierarchicalSparsePoolingGNN(cfg, stats)
        else:
            raise ValueError(f"Unknown hierarchical pooling type: {cfg.model.pooling_type}")
    else:
        raise ValueError(f"Unknown model name: {cfg.model.name}")
    
    
    return model
    
