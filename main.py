# driver script running requested experiment based on hydra config
import os
import wandb

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from src.data_processing.data_fetcher import get_data
from src.utils import set_seed, count_parameters, resolve_data_dir, resolve_model_dir
from src.train_eval.training import run_training
from src.models.model_factory import get_model

import warnings
import json


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

@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig) -> None:
    """Run one experiment from a Hydra config.

    Args:
        cfg: Resolved Hydra configuration containing dataset, model, logging,
            and optimization settings.

    Returns:
        None. Writes artifacts to disk and logs metrics to Weights & Biases.
    """
    print("Working Directory: ", os.getcwd())
    print("Hydra verison: ", hydra.__version__)

    # print config
    print(OmegaConf.to_yaml(cfg))
    print(cfg.logging.wandb_run_name)

    ### SETTINGS ###
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(cfg.experiment.seed)

    ### Move to data folder on local machine if needed ###
    is_local = torch.mps.is_available()
    if is_local and cfg.get("local_data_dir", None) is not None:
        os.chdir(cfg.local_data_dir)
        print("Changed working directory to local data dir: ", os.getcwd())


    ### BOOKKEEPING ###
    cfg = resolve_data_dir(cfg) # resolve logging_dir based on dataset settings
    cfg = resolve_model_dir(cfg)
    if not os.path.exists(cfg.logging.save_dir):
        os.makedirs(cfg.logging.save_dir)

    #### DATA ####
    (
    train_loader,
    val_loader,
    test_loader,
    stats,
    sg_stats
    ) = get_data(cfg)

    if sg_stats:
        with open(os.path.join(cfg.logging.save_dir, "subgraph_stats.json"), 'w') as f:
            json.dump(sg_stats, f, indent=4)

    #### Model ####
    model = get_model(cfg, stats).to(device)
    model_params = count_parameters(model)
    print("Model Parameters: ", model_params)

    ### Training ####
    # start wandb run
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    config_dict.update({'model_params': model_params,
                        "data_stats": stats})
    
    ## dump config_dict to file
    config_path = os.path.join(cfg.logging.save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    run = wandb.init(
        # Set the project where this run will be logged
        project=cfg.logging.wandb_project,
        name=f"{cfg.logging.wandb_run_name}",
        # Track hyperparameters and run metadata (convert Hydra DictConfig to plain dict)
        config=config_dict,
    )

    # initialize optimizer scheduler
    optimizer = torch.optim.AdamW(model.parameters(),
                                lr=cfg.dataset.optimization_params.learning_rate,
                                weight_decay=cfg.dataset.optimization_params.weight_decay)
    
    if cfg.dataset.optimization_params.scheduler == "ReduceLROnPlateau":
        default_monitor_mode = "min" if stats["task_type"] in {"regression", "multi_target_regression"} else "max"
        monitor_mode = cfg.dataset.optimization_params.get("monitor_mode", default_monitor_mode)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                            mode=monitor_mode,
                                                            factor=0.5,
                                                            min_lr=1e-6,
                                                            patience=40, 
                                                            verbose=True)
    elif cfg.dataset.optimization_params.scheduler == "CosineAnnealingLR":
        # init with warmup epochs with LinearLR, then CosineAnnealingLR
        warmup_epochs = cfg.dataset.optimization_params.get("warmup_epochs", 10)
        cosine_epochs = cfg.dataset.optimization_params.max_epochs - warmup_epochs
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
            ],
            milestones=[warmup_epochs]
        )
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.dataset.optimization_params.scheduler}")
    
    performances = run_training(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler, 
                        train_loader=train_loader, 
                        val_loader=val_loader,
                        test_loader=test_loader,
                        device=device,
                        num_epochs=cfg.dataset.optimization_params.max_epochs,
                        early_stop_patience=cfg.dataset.optimization_params.early_stop_patience,
                        output_dir=cfg.logging.save_dir,
                        num_classes=stats.get("num_classes") if stats is not None else None,
                        dataset_name=cfg.dataset.name,
                        task_type=stats.get("task_type") if stats is not None else None,
                        num_targets=stats.get("num_targets") if stats is not None else None,
                        target_names=stats.get("target_column_names") if stats is not None else None,
                        monitor_metric_name_override=cfg.dataset.optimization_params.get("monitor_metric", None),
                        monitor_mode_override=cfg.dataset.optimization_params.get("monitor_mode", None),
                            )
    
    torch.save(performances, os.path.join(cfg.logging.save_dir, "performances.pt"))
    

    # close wandb run
    run.finish()



if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    main()
