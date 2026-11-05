from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List
import time
import os
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import torch_geometric
from torch.nn import CosineEmbeddingLoss
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import wandb

from src.BREC.BRECDataset_v3 import BRECDataset, BRECDiskDataset
from src.data_processing.data_fetcher import get_subgraph_transform
from src.models.model_factory import get_model
from src.utils import resolve_model_dir, count_parameters

from src.data_processing.data_utils import (
    get_dataset_stats,
    compute_subgraph_statistics,
)

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch_geometric.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def build_pre_transform(cfg: DictConfig, seed: int):
    return get_subgraph_transform(cfg)


def build_dataset(cfg: DictConfig, seed: int) -> tuple[BRECDataset, float]:

    # continue building pre_transform
    pre_transform = build_pre_transform(cfg, seed)
    time_start = time.time()
    if cfg.dataset.get("on_disk", False):
        print("Using on-disk dataset...")
        dataset = BRECDiskDataset(
            name=cfg.dataset.name,
            root=cfg.dataset.root,
            pre_transform=pre_transform,
            force_reload=cfg.dataset.force_reload,
        )
    else:
        print("Using in-memory dataset...")

        if cfg.dataset.force_reload:
            print("Forcing dataset reload for in-memory dataset ...")
            root = Path(get_original_cwd()) / cfg.dataset.root
            # delete processed dir under root, since force reload doesnt always seem to work
            processed_dir = root / "BREC" / "processed"
            if processed_dir.exists() and processed_dir.is_dir():
                files_to_safe_delete = ['pre_transform.pt', 'pre_filter.pt', 'brec_v3.pt']
                for item in processed_dir.iterdir():
                    if item.name in files_to_safe_delete:
                        if item.name.endswith('.pt'):
                            print("Deleting file: ", item)
                            item.unlink()

        dataset = BRECDataset(
            name=cfg.dataset.name,
            root=cfg.dataset.root,
            pre_transform=pre_transform,
            force_reload=cfg.dataset.force_reload,
        )
    dataset_time = time.time() - time_start


    # get dataset stats
    dataset_stats = get_dataset_stats(dataset[0], cfg)
    if hasattr(dataset[0], "sg_nodes_orig") and hasattr(dataset[0], "sg_edges_orig"):
        subgraph_stats = compute_subgraph_statistics(dataset)
        print("Subgraph statistics: ", subgraph_stats)
    else:
        subgraph_stats = {}
        print("No original subgraph information found in dataset, skipping subgraph statistics.")

    return dataset, dataset_time, dataset_stats, subgraph_stats


def build_model(cfg: DictConfig, device: torch.device, dataset_stats: Dict) -> torch.nn.Module:
    # overwrite output dim in dataset stats
    dataset_stats["num_classes"] = cfg.model_runner.output_dim
    # keeping hidden dims and layers from datasets stats thou

    model = get_model(cfg, dataset_stats).to(device)
    return model


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DictConfig):
    sched_cfg = cfg.experiment.get("scheduler")
    if not sched_cfg:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=sched_cfg.get("mode", "min"),
        factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 10),
        min_lr=sched_cfg.get("min_lr", 1e-6),
        verbose=False,
    )


def compute_t_squared(
    dataset_slice, model: torch.nn.Module, device: torch.device, batch_size: int
) -> float:
    loader = DataLoader(dataset_slice, batch_size=batch_size, shuffle=False)
    pred_0, pred_1 = [], []
    with torch.no_grad():
        for data in loader:
            out = model(data.to(device)).detach()
            pred_0.append(out[0::2])
            pred_1.append(out[1::2])
    if not pred_0 or not pred_1:
        return 0.0
    X = torch.cat(pred_0, dim=0).T
    Y = torch.cat(pred_1, dim=0).T
    D = X - Y
    D_mean = torch.mean(D, dim=1, keepdim=True)
    cov = torch.cov(D)
    try:
        inv_cov = torch.linalg.pinv(cov)
    except (torch.linalg.LinAlgError, RuntimeError) as e:
        print("Error computing pseudo-inverse:", e)
        print("Data Object: ", data)
        # print("Failed Graph has {} subgraphs in {} batch components".format(data.num_subgraphs, data.batch.max().item()+1))
        print("X and Y are nan:", torch.isnan(X).any(), torch.isnan(Y).any())
        print("X and Y Norms: ", torch.norm(X), torch.norm(Y))
        print("D_mean is nan:", torch.isnan(D_mean).any())
        print("Cov is nan:", torch.isnan(cov).any())
        return float("nan")
    
    tsq = torch.mm(torch.mm(D_mean.T, inv_cov), D_mean)
    return float(tsq.item())


def start_wandb_run(
    cfg: DictConfig,
    seed: int,
    dataset_stats: Dict,
    subgraph_stats: Dict,
    model_params: int,
    dataset_time: float,
):
    if not cfg.logging.get("wandb", False):
        return None

    config_dict = OmegaConf.to_object(cfg)
    config_dict.update(
        {
            "dataset_stats": dataset_stats,
            "subgraph_stats": subgraph_stats,
            "model_params": model_params,
            "seed": seed,
            "dataset_time": dataset_time,
        }
    )
    run_name = cfg.logging.get(
        "wandb_run_name",
        f"brec_{cfg.model.name}_seed{seed}",
    )
    project = cfg.logging.get("wandb_project", "BREC")
    return wandb.init(project=project, name=run_name, config=config_dict)


def log_wandb_metrics(run, metrics: Dict, subgraph_stats: Dict):
    if run is None:
        return
    payload = {}
    overall = metrics.get("overall", {})
    for key, val in overall.items():
        payload[f"overall/{key}"] = val

    timings = metrics.get("timings", {})
    for key, val in timings.items():
        payload[f"timings/{key}"] = val

    for part_name, part_data in metrics.get("by_partition", {}).items():
        for key, val in part_data.items():
            if isinstance(val, (int, float)):
                payload[f"partition/{part_name}/{key}"] = val

    for key, val in subgraph_stats.items():
        payload[f"subgraph/{key}"] = val

    run.log(payload)
    run.summary.update(payload)


def evaluate_seed(
    cfg: DictConfig, dataset: BRECDataset, device: torch.device, seed: int, dataset_stats: Dict
) -> Dict:
    bench_cfg = cfg.benchmark
    part_lengths = sum(int(part.end - part.start) for part in bench_cfg.partitions)
    if part_lengths != bench_cfg.sample_pairs:
        raise ValueError(
            "Benchmark configuration mismatch: "
            f"sum(partition sizes)={part_lengths} but sample_pairs={bench_cfg.sample_pairs}"
        )

    num_relabel = bench_cfg.num_relabel
    sample_pairs = bench_cfg.sample_pairs
    threshold = cfg.evaluation.threshold
    epsilon_cmp = cfg.evaluation.get("epsilon_cmp", 1e-6)
    
    loss_func = CosineEmbeddingLoss(margin=cfg.experiment.margin)

    timings: Dict[str, float] = {
        "training_seconds": 0.0,
        "inference_seconds": 0.0,
        "evaluation_seconds": 0.0,
    }

    metrics: Dict = {
        "seed": seed,
        "overall": {
            "total_pairs": sample_pairs,
            "correct_pairs": 0,
            "accuracy": 0.0,
            "reliability_failures": 0,
            "threshold": threshold,
        },
        "by_partition": {},
        "correct_problem_ids": [],
        "timings": timings,
    }

    total_correct = 0
    fail_in_reliability = 0
    correct_ids: List[int] = []
    eval_start = time.time()

    partition_iter = tqdm(
        bench_cfg.partitions,
        desc=f"Partitions (seed={seed})",
        position=0,
    )
    for part in partition_iter:
        part_name = part.name
        part_correct = 0
        part_reliability_fail = 0
        part_correct_ids: List[int] = []
        part_train_time = 0.0
        part_infer_time = 0.0
        part_start = time.time()
        start_idx, end_idx = int(part.start), int(part.end)
        pair_iter = tqdm(
            range(start_idx, end_idx),
            desc=f"{part_name} (seed={seed})",
            leave=True,
            position=1,
        )
        for brec_id in pair_iter:
            model = build_model(cfg, device, dataset_stats)
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.experiment.learning_rate,
                weight_decay=cfg.experiment.weight_decay,
            )
            scheduler = build_scheduler(optimizer, cfg)

            dataset_traintest = dataset[
                brec_id * num_relabel * 2 : (brec_id + 1) * num_relabel * 2
            ]
            dataset_reliability = dataset[
                (brec_id + sample_pairs)
                * num_relabel
                * 2 : (brec_id + sample_pairs + 1)
                * num_relabel
                * 2
            ]

            model.train()
            train_start = time.time()
            for _ in range(cfg.experiment.epochs):
                loader = DataLoader(
                    dataset_traintest,
                    batch_size=cfg.experiment.batch_size,
                    shuffle=True,
                )
                loss_all = 0.0
                for data in loader:
                    optimizer.zero_grad()
                    pred = model(data.to(device))
                    target = torch.full(
                        (pred.shape[0] // 2,), -1.0, device=device, dtype=torch.float32
                    )
                    loss = loss_func(pred[0::2], pred[1::2], target)
                    loss.backward()
                    optimizer.step()
                    loss_all += (pred.shape[0] / 2) * loss.item()
                loss_all /= num_relabel
                if scheduler is not None:
                    scheduler.step(loss_all)
                if loss_all < cfg.experiment.loss_threshold:
                    break
            train_dur = time.time() - train_start
            timings["training_seconds"] += train_dur
            part_train_time += train_dur

            model.eval()
            infer_start = time.time()
            t_train = compute_t_squared(
                dataset_traintest, model, device, cfg.experiment.batch_size
            )
            t_reliability = compute_t_squared(
                dataset_reliability, model, device, cfg.experiment.batch_size
            )
            infer_dur = time.time() - infer_start
            timings["inference_seconds"] += infer_dur
            part_infer_time += infer_dur

            is_isomorphic = (
                t_train > threshold and abs(t_train - t_reliability) > epsilon_cmp
            )
            reliability_ok = t_reliability < threshold

            if is_isomorphic:
                total_correct += 1
                part_correct += 1
                correct_ids.append(brec_id)
                part_correct_ids.append(brec_id)

            if not reliability_ok:
                fail_in_reliability += 1
                part_reliability_fail += 1

            pair_iter.set_postfix(
                correct=part_correct,
                reliability_failures=part_reliability_fail,
            )
        pair_iter.close()

        total = end_idx - start_idx
        metrics["by_partition"][part_name] = {
            "total": total,
            "correct": part_correct,
            "accuracy": part_correct / total if total else 0.0,
            "reliability_failures": part_reliability_fail,
            "correct_problem_ids": part_correct_ids,
            "train_seconds": part_train_time,
            "inference_seconds": part_infer_time,
            "wall_seconds": time.time() - part_start,
        }

    accuracy = total_correct / sample_pairs if sample_pairs else 0.0
    metrics["overall"].update(
        {
            "correct_pairs": total_correct,
            "accuracy": accuracy,
            "reliability_failures": fail_in_reliability,
        }
    )
    metrics["correct_problem_ids"] = correct_ids
    timings["evaluation_seconds"] = time.time() - eval_start
    return metrics


def build_run_directory(cfg: DictConfig, seed: int) -> Path:
    base_dir = Path(get_original_cwd()) / cfg.logging.base_dir
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name

    cfg = resolve_model_dir(cfg)

    run_dir = base_dir / dataset_name / model_name / cfg.logging.clean_model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_artifacts(metrics: Dict, cfg: DictConfig, seed: int, run_dir: Path, subgraph_stats: Dict) -> None:
    metrics_path = run_dir / cfg.logging.metrics_filename
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    hyperparams = {
        "seed": seed,
        "dataset": OmegaConf.to_container(cfg.dataset, resolve=True),
        "model": OmegaConf.to_container(cfg.model, resolve=True),
        "model_runner": OmegaConf.to_container(cfg.model_runner, resolve=True),
        "experiment": OmegaConf.to_container(cfg.experiment, resolve=True),
        "benchmark": OmegaConf.to_container(cfg.benchmark, resolve=True),
        "evaluation": OmegaConf.to_container(cfg.evaluation, resolve=True),
    }
    hyper_path = run_dir / cfg.logging.hyperparams_filename
    with hyper_path.open("w", encoding="utf-8") as f:
        json.dump(hyperparams, f, indent=2)

    # dump subgraph stats
    subgraph_stats_path = run_dir / cfg.logging.subgraph_stats_filename
    with subgraph_stats_path.open("w", encoding="utf-8") as f:
        json.dump(subgraph_stats, f, indent=2)


def run_single_seed(cfg: DictConfig, seed: int) -> Dict:
    set_global_seed(seed)
    run_dir = build_run_directory(cfg, seed)
    device = resolve_device(cfg.experiment.device)
    dataset, dataset_time, dataset_stats, subgraph_stats = build_dataset(cfg, seed)
    print("Dataset Stats: ", dataset_stats)
    template_model = get_model(cfg, dataset_stats)
    model_params = count_parameters(template_model)
    del template_model

    wandb_run = start_wandb_run(cfg, seed, dataset_stats, subgraph_stats, model_params, dataset_time)
    metrics = evaluate_seed(cfg, dataset, device, seed, dataset_stats)
    metrics.setdefault("timings", {})
    metrics["timings"]["dataset_seconds"] = dataset_time
    metrics["timings"]["total_seconds"] = (
        metrics["timings"].get("evaluation_seconds", 0.0) + dataset_time
    )

    save_artifacts(metrics, cfg, seed, run_dir, subgraph_stats)
    log_wandb_metrics(wandb_run, metrics, subgraph_stats)
    if wandb_run is not None:
        wandb_run.finish()
    return metrics


@hydra.main(version_base=None, config_path="conf", config_name="brec")
def main(cfg: DictConfig) -> None:

    ### Move to data folder on local machine if needed ###
    is_local = torch.mps.is_available()
    print("Is local machine with MPS available: ", is_local)
    if is_local and cfg.get("local_data_dir", None) is not None:
        os.chdir(cfg.local_data_dir)
        print("Changed working directory to local data dir: ", os.getcwd())

    seed_values = [cfg.experiment.seed]
    seed_bar = tqdm(seed_values, desc="Seeds", position=0)
    for seed in seed_bar:
        metrics = run_single_seed(cfg, int(seed))
        seed_bar.set_postfix(accuracy=f"{metrics['overall']['accuracy']:.3f}")
        seed_bar.write(
            f"[Seed {seed}] Correct {metrics['overall']['correct_pairs']} "
            f"/ {metrics['overall']['total_pairs']}"
        )
    seed_bar.close()


if __name__ == "__main__":
    main()
