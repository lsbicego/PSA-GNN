import torch
import numpy as np

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def resolve_data_dir(cfg):
    if cfg.dataset.name == "ZINC":
        subset_str = "subset" if cfg.dataset.subset else "full"
        cfg.logging.save_dir = cfg.logging.save_dir.replace("ZINC", f"ZINC_{subset_str}")
    return cfg

def resolve_model_dir(cfg):
    if cfg.model.name in ["pool_gnn", "ip_gnn", "psa_gnn"]:
        sg_method = cfg.subgraph_policy.method
        marking_level = cfg.subgraph_policy.get("marking_level", "no_marking")
        khop = str(cfg.subgraph_policy.get("k_hop", 1))
        cfg.logging.save_dir = cfg.logging.save_dir.replace(cfg.model.name, f"{marking_level}/{cfg.model.name}_{sg_method}_{marking_level}_k={khop}")
        cfg.logging.clean_model_name = f"{cfg.model.name}_{sg_method}_{marking_level}_k={khop}"
    else:
        cfg.logging.clean_model_name = cfg.model.name
    return cfg
