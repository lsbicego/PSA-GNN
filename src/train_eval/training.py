from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import wandb
from torch import nn
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from .training_helpers import (
    ClassificationMeter,
    MultiLabelMeter,
    MultiTargetRegressionMeter,
    RegressionMeter,
    TaskInfo,
    create_classification_plots,
    create_multilabel_plots,
    create_multi_target_regression_plots,
    create_regression_plots,
    gather_predictions,
    get_batch_size,
    infer_task_type,
    model_size_bytes,
    save_disaggregated_metrics,
    save_metrics_json,
    select_monitor_metric,
)


def _compute_loss_and_predictions(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    task_info: TaskInfo,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the loss and metric-ready predictions for one batch."""
    if task_info.task_type in {"regression", "multi_target_regression"}:
        targets = targets.float()
        if outputs.shape != targets.shape:
            targets = targets.view_as(outputs)
        loss = F.l1_loss(outputs, targets)
        return loss, outputs, targets

    if task_info.task_type == "multilabel":
        targets = targets.float()
        outputs = outputs.view(-1, task_info.num_classes or outputs.shape[-1])
        targets = targets.view_as(outputs)
        loss = F.binary_cross_entropy_with_logits(outputs, targets)
        probs = torch.sigmoid(outputs)
        return loss, probs, targets

    if task_info.num_classes is None:
        raise ValueError("Classification task requires 'num_classes' in TaskInfo.")

    offset = task_info.label_offset if task_info.label_offset is not None else 0
    num_classes = task_info.num_classes or 2

    if num_classes <= 2 and (outputs.ndim == 1 or outputs.shape[-1] == 1):
        logits = outputs.view(-1)
        targets = targets.view(-1).float() - float(offset)
        targets = (targets > 0).float()
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        preds = (torch.sigmoid(logits) >= 0.5).long()
        true_labels = targets.long()
    else:
        logits = outputs if outputs.ndim > 1 else outputs.unsqueeze(-1)
        true_labels = targets.view(-1).long() - int(offset)
        true_labels = torch.clamp(true_labels, min=0, max=num_classes - 1)
        loss = F.cross_entropy(logits, true_labels)
        preds = logits.argmax(dim=-1)

    return loss, preds, true_labels


def _run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    task_info: TaskInfo,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 2.0,
) -> Tuple[float, Dict[str, float], Dict[str, float], float]:
    """Run one training or evaluation epoch over a loader."""
    train_mode = optimizer is not None
    model.train(train_mode)
    if task_info.task_type == "regression":
        meter = RegressionMeter()
    elif task_info.task_type == "multi_target_regression":
        meter = MultiTargetRegressionMeter(task_info.target_names)
    elif task_info.task_type == "multilabel":
        meter = MultiLabelMeter(task_info.num_classes or 1)
    else:
        meter = ClassificationMeter(task_info.num_classes or 2)

    total_loss = 0.0
    total_examples = 0
    grad_norms = []
    start_time = time.perf_counter()
    loop = tqdm(loader, leave=True, desc="train" if train_mode else "eval")

    def _postfix_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
        return {k: v for k, v in metrics.items() if not k.startswith("ap_class_") and not k.startswith("f1_class_")}

    for batch in loop:
        batch = batch.to(device, non_blocking=True) if hasattr(batch, "to") else batch
        targets = getattr(batch, "y", None)
        if targets is None:
            raise ValueError("Batch does not contain targets 'y'.")
        targets = targets.to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            outputs = model(batch)
            loss, preds_for_metrics, targets_for_metrics = _compute_loss_and_predictions(outputs, targets, task_info)
            if hasattr(model, "reg_loss"):
                loss = loss + model.reg_loss

        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(clip_grad_norm_(model.parameters(), grad_clip))
            optimizer.step()
            grad_norms.append(grad_norm)

        meter.update(preds_for_metrics, targets_for_metrics)
        batch_size = get_batch_size(batch, targets_for_metrics)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

        snapshot = {"loss": total_loss / max(total_examples, 1)}
        snapshot.update(_postfix_metrics(meter.compute()))
        if train_mode and grad_norms:
            snapshot["grad_norm"] = grad_norms[-1]
        loop.set_postfix({k: f"{v:.4f}" for k, v in snapshot.items() if isinstance(v, (int, float))})

    epoch_time = time.perf_counter() - start_time
    avg_loss = total_loss / max(total_examples, 1)
    metrics = meter.compute()
    grad_stats = {}
    if grad_norms:
        grad_stats = {
            "grad_norm_mean": sum(grad_norms) / len(grad_norms),
            "grad_norm_max": max(grad_norms),
            "grad_norm_min": min(grad_norms),
        }
    return avg_loss, metrics, grad_stats, epoch_time


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    num_epochs: int,
    early_stop_patience: int,
    output_dir: Optional[Path] = None,
    grad_clip: float = 5.0,
    num_classes: Optional[int] = None,
    dataset_name: Optional[str] = None,
    task_type: Optional[str] = None,
    num_targets: Optional[int] = None,
    target_names: Optional[List[str]] = None,
    monitor_metric_name_override: Optional[str] = None,
    monitor_mode_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Train a model, select the best checkpoint, and export evaluation artifacts."""
    task_info = infer_task_type(
        train_loader,
        provided_task=task_type,
        provided_num_classes=num_classes,
        provided_num_targets=num_targets,
    )
    task_info.num_classes = num_classes if num_classes is not None else task_info.num_classes
    task_info.num_targets = num_targets if num_targets is not None else task_info.num_targets
    task_info.target_names = target_names
    print("Configured task info:", task_info)
    model.to(device)
    wandb_dir = Path(wandb.run.dir) if wandb.run is not None else Path.cwd()
    if output_dir is None:
        output_dir = wandb_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_root = output_dir / "plots"
    checkpoint_root = output_dir / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    dataset_name_lower = dataset_name.lower() if dataset_name else None
    metadata_fields: List[str] = []
    if dataset_name_lower == "omol":
        metadata_fields = ["domain", "is_connected"]
    elif dataset_name_lower == "sat":
        metadata_fields = ["hardness", "category"]
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_metric: Optional[float] = None
    best_test_metric: Optional[float] = None
    best_epoch = 0
    epochs_without_improvement = 0
    default_monitor_mode = "min" if task_info.task_type in {"regression", "multi_target_regression"} else "max"
    monitor_mode = monitor_mode_override.lower() if monitor_mode_override is not None else default_monitor_mode
    if monitor_mode not in {"min", "max"}:
        raise ValueError(f"Invalid monitor mode '{monitor_mode}'. Expected 'min' or 'max'.")
    monitor_metric_name: Optional[str] = None

    history: Dict[str, list] = {
        "train": [],
        "val": [],
        "test": [],
        "loss_train": [],
        "loss_val": [],
        "loss_test": [],
        "lr": [],
        "grad_norm": [],
        "train_epoch_times": [],
        "val_eval_times": [],
        "test_eval_times": [],
    }

    training_start = time.perf_counter()
    for epoch in range(1, num_epochs + 1):
        train_loss, train_metrics, grad_stats, train_time = _run_epoch(model, train_loader, device, task_info, optimizer=optimizer, grad_clip=grad_clip)
        val_loss, val_metrics, _, val_time = _run_epoch(model, val_loader, device, task_info, optimizer=None, grad_clip=grad_clip)
        test_loss, test_metrics, _, test_time = _run_epoch(model, test_loader, device, task_info, optimizer=None, grad_clip=grad_clip)

        monitor_metric_name, monitor_metric_value = select_monitor_metric(task_info, val_metrics, preferred_metric=monitor_metric_name_override)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitor_metric_value)
            else:
                scheduler.step()

        improved = False
        if best_metric is None:
            improved = True
        elif monitor_mode == "min":
            improved = monitor_metric_value < best_metric - 1e-6
        else:
            improved = monitor_metric_value > best_metric + 1e-6

        if improved:
            best_metric = monitor_metric_value
            _, current_test_metric_value = select_monitor_metric(task_info, test_metrics, preferred_metric=monitor_metric_name_override)
            best_test_metric = current_test_metric_value
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, checkpoint_root / "best_model.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        current_lr = optimizer.param_groups[0]["lr"]
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        history["test"].append(test_metrics)
        history["loss_train"].append(train_loss)
        history["loss_val"].append(val_loss)
        history["loss_test"].append(test_loss)
        history["lr"].append(current_lr)
        history["grad_norm"].append(grad_stats)
        history["train_epoch_times"].append(train_time)
        history["val_eval_times"].append(val_time)
        history["test_eval_times"].append(test_time)

        log_payload = {
            "epoch": epoch,
            "lr": current_lr,
            "monitor_metric": monitor_metric_value,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "test/loss": test_loss,
            "time/train_epoch_sec": train_time,
            "time/val_eval_sec": val_time,
            "time/test_eval_sec": test_time,
        }
        if best_test_metric is not None:
            log_payload["charts/best_test_metric"] = best_test_metric
        for split, metrics in zip(("train", "val", "test"), (train_metrics, val_metrics, test_metrics)):
            for name, value in metrics.items():
                log_payload[f"{split}/{name}"] = value
        for key, value in grad_stats.items():
            log_payload[f"gradients/{key}"] = value
        wandb.log(log_payload, step=epoch)

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            break

    total_training_time = time.perf_counter() - training_start

    if best_state is not None:
        model.load_state_dict(best_state)

    final_train_loss, final_train_metrics, _, final_train_time = _run_epoch(model, train_loader, device, task_info, optimizer=None, grad_clip=grad_clip)
    final_val_loss, final_val_metrics, _, final_val_time = _run_epoch(model, val_loader, device, task_info, optimizer=None, grad_clip=grad_clip)
    final_test_loss, final_test_metrics, _, final_test_time = _run_epoch(model, test_loader, device, task_info, optimizer=None, grad_clip=grad_clip)

    final_monitor_name, final_monitor_value = select_monitor_metric(task_info, final_val_metrics, preferred_metric=monitor_metric_name_override)

    plots_root.mkdir(parents=True, exist_ok=True)
    prediction_outputs = {
        "train": gather_predictions(model, train_loader, device, task_info, metadata_fields=metadata_fields),
        "val": gather_predictions(model, val_loader, device, task_info, metadata_fields=metadata_fields),
        "test": gather_predictions(model, test_loader, device, task_info, metadata_fields=metadata_fields),
    }

    plots_registry: Dict[str, List[str]] = {}
    extra_metrics_by_split: Dict[str, Dict[str, float]] = {}
    for split_name, data in prediction_outputs.items():
        split_dir = plots_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        targets_tensor = data.get("targets", torch.empty(0))
        preds_tensor = data.get("predictions", torch.empty(0))

        if task_info.task_type == "regression":
            saved_paths = create_regression_plots(targets_tensor, preds_tensor, split_name, split_dir)
            plots_registry[split_name] = [str(path) for path in saved_paths]
        elif task_info.task_type == "multi_target_regression":
            saved_paths = create_multi_target_regression_plots(
                targets_tensor,
                preds_tensor,
                split_name,
                split_dir,
                target_names=task_info.target_names,
            )
            plots_registry[split_name] = [str(path) for path in saved_paths]
        elif task_info.task_type == "multilabel":
            probs_tensor = data.get("probabilities")
            saved_paths = create_multilabel_plots(targets_tensor, probs_tensor, split_name, split_dir, task_info, top_k=5)
            plots_registry[split_name] = [str(path) for path in saved_paths]
        else:
            probs_tensor = data.get("probabilities")
            try:
                saved_paths, extras = create_classification_plots(
                    targets_tensor,
                    preds_tensor,
                    probs_tensor,
                    split_name,
                    split_dir,
                    task_info,
                )
                plots_registry[split_name] = [str(path) for path in saved_paths]
                if extras:
                    extra_metrics_by_split[split_name] = extras
            except Exception as e:
                print(f"Error creating classification plots for split '{split_name}': {e}")
                plots_registry[split_name] = []

    for split_name, extras in extra_metrics_by_split.items():
        if split_name == "train":
            final_train_metrics.update(extras)
        elif split_name == "val":
            final_val_metrics.update(extras)
        elif split_name == "test":
            final_test_metrics.update(extras)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = model_size_bytes(model)

    performance: Dict[str, Any] = {
        "task": task_info.task_type,
        "num_targets": task_info.num_targets,
        "num_classes": task_info.num_classes,
        "best_epoch": best_epoch,
        "best_val_metric_name": monitor_metric_name or final_monitor_name,
        "best_val_metric": best_metric if best_metric is not None else final_monitor_value,
        "final_val_metric_name": final_monitor_name,
        "final_val_metric": final_monitor_value,
        "history": history,
        "train_final": {"loss": final_train_loss, **final_train_metrics},
        "val_final": {"loss": final_val_loss, **final_val_metrics},
        "test_final": {"loss": final_test_loss, **final_test_metrics},
        "timings": {
            "total_training_time": total_training_time,
            "average_epoch_time": sum(history["train_epoch_times"]) / max(len(history["train_epoch_times"]), 1),
            "train_epoch_times": history["train_epoch_times"],
            "val_eval_times": history["val_eval_times"],
            "test_eval_times": history["test_eval_times"],
            "final_train_eval_time": final_train_time,
            "final_val_eval_time": final_val_time,
            "final_test_eval_time": final_test_time,
            "final_inference_time": final_train_time + final_val_time + final_test_time,
        },
        "model": {
            "num_parameters": num_params,
            "size_bytes": size_bytes,
            "size_megabytes": size_bytes / (1024 ** 2),
            "checkpoint_dir": str(checkpoint_root),
        },
    }

    metrics_path = save_metrics_json(performance, output_dir)
    disagg_paths = save_disaggregated_metrics(dataset_name_lower, prediction_outputs, task_info, output_dir)
    performance["artifacts"] = {
        "metrics_json": str(metrics_path),
        "plots_dir": str(plots_root),
        "plots": plots_registry,
        "disaggregated_metrics": [str(path) for path in disagg_paths],
    }

    if wandb.run is not None:
        image_payload = {}
        for split_name, saved in plots_registry.items():
            for plot_path in saved:
                image_payload[f"plots/{split_name}/{Path(plot_path).stem}"] = wandb.Image(plot_path)
        if image_payload:
            wandb.log(image_payload)

    wandb.log(
        {
            "final/train_loss": final_train_loss,
            **{f"final/train_{k}": v for k, v in final_train_metrics.items()},
            "final/val_loss": final_val_loss,
            **{f"final/val_{k}": v for k, v in final_val_metrics.items()},
            "final/test_loss": final_test_loss,
            **{f"final/test_{k}": v for k, v in final_test_metrics.items()},
            "summary/best_epoch": best_epoch,
        }
    )

    return performance
