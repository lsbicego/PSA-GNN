from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch import nn
from torch_scatter import scatter


@dataclass
class TaskInfo:
    """Minimal task description used to choose losses and metrics."""

    task_type: str
    num_targets: int
    num_classes: Optional[int] = None
    label_offset: int = 0
    target_names: Optional[List[str]] = None


class RegressionMeter:
    def __init__(self) -> None:
        self.abs_error = 0.0
        self.squared_error = 0.0
        self.pct_error = 0.0
        self.num_elements = 0
        self.non_zero_targets = 0
        self.sum_targets = 0.0
        self.sum_targets_sq = 0.0

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach().view(-1)
        targets = targets.detach().view(-1)
        diff = preds - targets
        self.abs_error += torch.abs(diff).sum()
        self.squared_error += torch.square(diff).sum()
        self.sum_targets += targets.sum().item()
        self.sum_targets_sq += torch.square(targets).sum().item()
        non_zero_mask = torch.abs(targets) > 1e-8
        if non_zero_mask.any():
            self.pct_error += torch.abs(diff[non_zero_mask] / targets[non_zero_mask]).sum()
            self.non_zero_targets += int(non_zero_mask.sum().item())
        self.num_elements += targets.numel()

    def compute(self) -> Dict[str, float]:
        if self.num_elements == 0:
            return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan")}
        mae = float((self.abs_error / self.num_elements).item())
        rmse = math.sqrt(float(self.squared_error / self.num_elements))
        mape = float(self.pct_error / self.non_zero_targets) if self.non_zero_targets > 0 else float("nan")
        mean_y = self.sum_targets / self.num_elements
        sst = self.sum_targets_sq - (self.num_elements * mean_y * mean_y)
        r2 = 1.0 - (float(self.squared_error) / sst) if sst > 0 else float("nan")
        return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


class MultiTargetRegressionMeter:
    def __init__(self, target_names: Optional[List[str]] = None) -> None:
        self.preds: List[torch.Tensor] = []
        self.targets: List[torch.Tensor] = []
        self.target_names = target_names

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.preds.append(preds.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        if not self.targets:
            return {}
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)
        assert preds.shape == targets.shape
        errors = torch.abs(preds - targets)
        sq_errors = torch.square(preds - targets)
        mae_per_target = errors.mean(dim=0)
        rmse_per_target = torch.sqrt(sq_errors.mean(dim=0))
        names = self.target_names or [f"target_{i}" for i in range(mae_per_target.numel())]
        metrics: Dict[str, float] = {}
        for idx, name in enumerate(names):
            metrics[f"mae_{name}"] = float(mae_per_target[idx].item())
            metrics[f"rmse_{name}"] = float(rmse_per_target[idx].item())
        metrics["mae_mean"] = float(mae_per_target.mean().item())
        metrics["rmse_mean"] = float(rmse_per_target.mean().item())
        return metrics


class ClassificationMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.correct = 0
        self.total = 0
        if num_classes <= 2:
            self.tp = 0
            self.fp = 0
            self.fn = 0
            self.tn = 0
        else:
            self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds = preds.detach().view(-1).long()
        targets = targets.detach().view(-1).long()
        self.correct += int((preds == targets).sum().item())
        self.total += int(targets.numel())
        if self.num_classes <= 2:
            self.tp += int(((preds == 1) & (targets == 1)).sum().item())
            self.tn += int(((preds == 0) & (targets == 0)).sum().item())
            self.fp += int(((preds == 1) & (targets == 0)).sum().item())
            self.fn += int(((preds == 0) & (targets == 1)).sum().item())
        else:
            for cls in range(self.num_classes):
                mask = targets == cls
                if mask.any():
                    cls_preds = preds[mask]
                    values, counts = torch.unique(cls_preds, return_counts=True)
                    for value, count in zip(values.tolist(), counts.tolist()):
                        self.confusion[cls, value] += count

    def compute(self) -> Dict[str, float]:
        if self.total == 0:
            return {"accuracy": float("nan")}
        metrics: Dict[str, float] = {"accuracy": self.correct / self.total}
        if self.num_classes <= 2:
            precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
            recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
            f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            metrics.update({"precision": precision, "recall": recall, "f1": f1})
        else:
            per_class_f1 = []
            for cls in range(self.num_classes):
                tp = self.confusion[cls, cls].item()
                fp = (self.confusion[:, cls].sum() - tp).item()
                fn = (self.confusion[cls, :].sum() - tp).item()
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                if (precision + recall) > 0:
                    per_class_f1.append((2.0 * precision * recall) / (precision + recall))
            if per_class_f1:
                metrics["macro_f1"] = sum(per_class_f1) / len(per_class_f1)
        return metrics


class MultiLabelMeter:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.targets: List[torch.Tensor] = []
        self.probs: List[torch.Tensor] = []

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        self.probs.append(probs.detach().cpu())
        self.targets.append(targets.detach().cpu())

    def compute(self) -> Dict[str, float]:
        if not self.targets:
            return {}
        y_true = torch.cat(self.targets, dim=0).numpy()
        y_prob = torch.cat(self.probs, dim=0).numpy()
        results: Dict[str, float] = {}
        try:
            results["ap_macro"] = float(average_precision_score(y_true, y_prob, average="macro"))
        except Exception:
            pass
        try:
            results["ap_micro"] = float(average_precision_score(y_true, y_prob, average="micro"))
        except Exception:
            pass
        preds = (y_prob >= 0.5).astype(int)
        tp = (preds * y_true).sum(axis=0)
        fp = (preds * (1 - y_true)).sum(axis=0)
        fn = ((1 - preds) * y_true).sum(axis=0)
        precision = tp / np.maximum(tp + fp, 1e-9)
        recall = tp / np.maximum(tp + fn, 1e-9)
        f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
        results["f1_macro"] = float(np.mean(f1))
        results["precision_macro"] = float(np.mean(precision))
        results["recall_macro"] = float(np.mean(recall))
        results["precision_micro"] = float(np.sum(tp) / max(np.sum(tp + fp), 1e-9))
        results["recall_micro"] = float(np.sum(tp) / max(np.sum(tp + fn), 1e-9))
        denom = results["precision_micro"] + results["recall_micro"]
        results["f1_micro"] = float(2 * results["precision_micro"] * results["recall_micro"] / denom) if denom > 0 else 0.0
        return results


def infer_task_type(
    train_loader,
    provided_task: Optional[str] = None,
    provided_num_classes: Optional[int] = None,
    provided_num_targets: Optional[int] = None,
) -> TaskInfo:
    sample_batch = next(iter(train_loader))
    targets = getattr(sample_batch, "y", None)
    if targets is None:
        raise ValueError("Expected batch to contain attribute 'y' for targets.")
    target_tensor = targets.detach().cpu()
    num_targets = target_tensor.shape[-1] if target_tensor.ndim > 1 else 1

    if provided_task is not None:
        task_lower = provided_task.lower()
        if "multi" in task_lower and "regress" in task_lower:
            return TaskInfo("multi_target_regression", num_targets=provided_num_targets or num_targets)
        if "regress" in task_lower:
            return TaskInfo("regression", num_targets=provided_num_targets or num_targets)
        if task_lower in {"classification", "binary_classification"}:
            num_classes = provided_num_classes or getattr(getattr(train_loader, "dataset", None), "num_classes", None)
            return TaskInfo("classification", num_targets=1, num_classes=num_classes)
        if task_lower in {"multilabel", "multilabel_classification", "multi_label_classification"}:
            num_classes = provided_num_classes or num_targets
            return TaskInfo("multilabel", num_targets=num_targets, num_classes=num_classes)

    unique_values = torch.unique(target_tensor.view(-1))
    dataset_num_classes = getattr(getattr(train_loader, "dataset", None), "num_classes", None)
    if dataset_num_classes is not None:
        num_classes = int(dataset_num_classes)
        min_value = int(unique_values.min().item()) if unique_values.numel() > 0 else 0
        return TaskInfo("classification", num_targets=1, num_classes=num_classes, label_offset=min_value)

    if target_tensor.dtype in (torch.long, torch.int32, torch.int64):
        num_classes = unique_values.numel()
        min_value = int(unique_values.min().item()) if unique_values.numel() > 0 else 0
        return TaskInfo("classification", num_targets=1, num_classes=int(num_classes), label_offset=min_value)

    if unique_values.numel() <= 20:
        approx_integers = torch.allclose(unique_values, unique_values.round(), atol=1e-4)
        if approx_integers:
            min_value = int(unique_values.min().item()) if unique_values.numel() > 0 else 0
            shifted = unique_values - min_value
            if shifted.min().item() >= -1e-6 and shifted.max().item() >= -1e-6:
                num_classes = int(shifted.max().item()) + 1
                return TaskInfo("classification", num_targets=1, num_classes=num_classes, label_offset=min_value)

    return TaskInfo("regression", num_targets=num_targets)


def get_batch_size(batch: Any, targets: torch.Tensor) -> int:
    if hasattr(batch, "num_graphs"):
        return int(batch.num_graphs)
    if targets.ndim > 0:
        return int(targets.shape[0])
    return 1


def select_monitor_metric(
    task_info: TaskInfo,
    metrics: Dict[str, float],
    preferred_metric: Optional[str] = None,
) -> Tuple[str, float]:
    if preferred_metric is not None:
        metric_value = metrics.get(preferred_metric)
        if metric_value is None:
            available = ", ".join(sorted(metrics.keys()))
            raise ValueError(
                f"Configured monitor metric '{preferred_metric}' is not available. "
                f"Available metrics: {available}"
            )
        return preferred_metric, metric_value

    if task_info.task_type == "regression":
        metric_name = "mae"
    elif task_info.task_type == "multi_target_regression":
        metric_name = "mae_mean"
    elif task_info.task_type == "multilabel":
        metric_name = "ap_macro" if "ap_macro" in metrics else "ap_micro"
    else:
        metric_name = "f1" if (task_info.num_classes or 0) <= 2 and "f1" in metrics else "macro_f1"
        if metric_name not in metrics:
            metric_name = "accuracy"
    metric_value = metrics.get(metric_name)
    if metric_value is None:
        raise ValueError(f"Metric '{metric_name}' not available for monitoring.")
    return metric_name, metric_value


def model_size_bytes(model: nn.Module) -> int:
    total = 0
    for param in model.parameters():
        total += param.nelement() * param.element_size()
    return total


def _normalize_metadata_values(value: Any, num_graphs: int) -> List[Any]:
    if num_graphs == 0:
        return []
    if value is None:
        return [None] * num_graphs
    if isinstance(value, (list, tuple)):
        values = list(value)
    elif torch.is_tensor(value):
        values = value.detach().cpu().view(-1).tolist()
    else:
        values = [value]
    if len(values) == num_graphs:
        return values
    if len(values) > num_graphs:
        return values[:num_graphs]
    if not values:
        return [None] * num_graphs
    return values + [values[-1]] * (num_graphs - len(values))


def _select_positive_probs(probabilities: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if probabilities is None:
        return None
    probs = probabilities.detach().cpu()
    if probs.ndim == 1:
        return probs
    if probs.ndim == 2:
        if probs.shape[1] == 1:
            return probs.view(-1)
        positive_idx = min(1, probs.shape[1] - 1)
        return probs[:, positive_idx]
    return probs.view(-1)


def gather_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
    task_info: TaskInfo,
    metadata_fields: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    model.eval()
    preds: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []
    probs: List[torch.Tensor] = []
    metadata_store: Dict[str, List[Any]] = {field: [] for field in (metadata_fields or [])}

    num_classes = task_info.num_classes or 1
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device) if hasattr(batch, "to") else batch
            targets = getattr(batch, "y", None)
            if targets is None:
                raise ValueError("Batch does not contain targets 'y'.")
            targets = targets.to(device)
            outputs = model(batch)

            if task_info.task_type in {"regression", "multi_target_regression"}:
                outputs = outputs.view(-1, task_info.num_targets)
                tgt = targets.view_as(outputs).float()
                preds.append(outputs.detach().cpu())
                targets_all.append(tgt.detach().cpu())
            elif task_info.task_type == "multilabel":
                logits = outputs.view(-1, num_classes)
                prob = torch.sigmoid(logits)
                tgt = targets.view_as(logits).float()
                preds.append(prob.detach().cpu())
                probs.append(prob.detach().cpu())
                targets_all.append(tgt.detach().cpu())
            else:
                offset = task_info.label_offset or 0
                logits = outputs if outputs.ndim > 1 else outputs.unsqueeze(-1)
                tgt = targets.view(-1).long() - int(offset)
                tgt = torch.clamp(tgt, min=0, max=num_classes - 1)
                targets_all.append(tgt.detach().cpu())

                if num_classes <= 2 and logits.shape[-1] == 1:
                    logits = logits.view(-1)
                    prob = torch.sigmoid(logits)
                    pred = (prob >= 0.5).long()
                    preds.append(pred.detach().cpu())
                    probs.append(prob.detach().cpu().unsqueeze(-1))
                else:
                    prob = torch.softmax(logits, dim=-1)
                    pred = torch.argmax(prob, dim=-1)
                    preds.append(pred.detach().cpu())
                    probs.append(prob.detach().cpu())

            if metadata_store:
                num_graphs = get_batch_size(batch, targets)
                for field in metadata_store:
                    value = getattr(batch, field, None)
                    metadata_store[field].extend(_normalize_metadata_values(value, num_graphs))

    gathered: Dict[str, torch.Tensor] = {}
    if targets_all:
        gathered["targets"] = torch.cat(targets_all, dim=0)
    if preds:
        gathered["predictions"] = torch.cat(preds, dim=0)
    if probs and task_info.task_type in {"classification", "multilabel"}:
        gathered["probabilities"] = torch.cat(probs, dim=0)
    if metadata_store:
        gathered["metadata"] = metadata_store
    return gathered


def save_metrics_json(performance: Dict[str, Any], output_dir: Path) -> Path:
    timings = performance.get("timings", {})
    summary = {
        "best_epoch": performance.get("best_epoch"),
        "best_val_metric_name": performance.get("best_val_metric_name"),
        "best_val_metric": performance.get("best_val_metric"),
        "train": performance.get("train_final", {}),
        "val": performance.get("val_final", {}),
        "test": performance.get("test_final", {}),
        "timings": {
            "total_training_time": timings.get("total_training_time"),
            "average_epoch_time": timings.get("average_epoch_time"),
            "final_inference_time": timings.get("final_inference_time"),
        },
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return metrics_path


def create_regression_plots(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    split: str,
    output_dir: Path,
) -> List[Path]:
    if targets.numel() == 0 or predictions.numel() == 0:
        return []
    y_true = targets.view(-1).numpy()
    y_pred = predictions.view(-1).numpy()
    plots: List[Path] = []

    def _hexbin_plot(y_a: np.ndarray, y_b: np.ndarray, path: Path, title: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        hb = ax.hexbin(y_a, y_b, gridsize=60, cmap="viridis")
        lim_min = min(np.min(y_a), np.min(y_b))
        lim_max = max(np.max(y_a), np.max(y_b))
        delta = (lim_max - lim_min) * 0.05 if lim_max != lim_min else 1.0
        ax.plot([lim_min - delta, lim_max + delta], [lim_min - delta, lim_max + delta], "--", color="crimson")
        ax.set_xlabel("True values")
        ax.set_ylabel("Predicted values")
        ax.set_title(title)
        cbar = fig.colorbar(hb, ax=ax)
        cbar.set_label("Counts")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)

    overall_path = output_dir / f"{split}_true_vs_pred_hexbin.png"
    _hexbin_plot(y_true, y_pred, overall_path, f"{split.title()} | Predictions vs. Targets")
    plots.append(overall_path)

    median = np.median(y_true)
    std = np.std(y_true)
    if std > 0:
        mask = (y_true >= median - std) & (y_true <= median + std)
        mask &= (y_pred >= median - std) & (y_pred <= median + std)
        if np.count_nonzero(mask) > 10:
            zoom_path = output_dir / f"{split}_true_vs_pred_hexbin_zoom.png"
            _hexbin_plot(y_true[mask], y_pred[mask], zoom_path, f"{split.title()} | Predictions vs. Targets (median ±1σ)")
            plots.append(zoom_path)

    return plots


def create_multi_target_regression_plots(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    split: str,
    output_dir: Path,
    target_names: Optional[List[str]] = None,
    max_plots: int = 5,
) -> List[Path]:
    if targets.numel() == 0 or predictions.numel() == 0:
        return []
    y_true = targets.detach().cpu()
    y_pred = predictions.detach().cpu()
    num_targets = y_true.shape[1] if y_true.ndim > 1 else 1
    names = target_names or [f"target_{i}" for i in range(num_targets)]
    plots: List[Path] = []
    for idx in range(min(num_targets, max_plots)):
        name = names[idx] if idx < len(names) else f"target_{idx}"
        subdir = output_dir / f"{split}_{name}"
        subdir.mkdir(parents=True, exist_ok=True)
        plots.extend(
            create_regression_plots(
                targets=y_true[:, idx],
                predictions=y_pred[:, idx],
                split=f"{split}_{name}",
                output_dir=subdir,
            )
        )
    return plots


def create_classification_plots(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    probabilities: Optional[torch.Tensor],
    split: str,
    output_dir: Path,
    task_info: TaskInfo,
) -> Tuple[List[Path], Dict[str, float]]:
    if targets.numel() == 0 or predictions.numel() == 0:
        return [], {}
    y_true = targets.view(-1).numpy()
    y_pred = predictions.view(-1).numpy()
    num_classes = task_info.num_classes or int(np.max(y_true) + 1)
    classes = np.arange(num_classes)
    plots: List[Path] = []
    extras: Dict[str, float] = {}

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(f"{split.title()} | Confusion Matrix")
    fig.tight_layout()
    cm_path = output_dir / f"{split}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=200)
    plt.close(fig)
    plots.append(cm_path)

    if probabilities is not None and probabilities.numel() > 0 and num_classes <= 2:
        prob_tensor = _select_positive_probs(probabilities)
        prob_np = prob_tensor.numpy() if prob_tensor is not None else None
        if prob_np is not None:
            try:
                fpr, tpr, _ = roc_curve(y_true, prob_np)
                roc_auc = auc(fpr, tpr)
                extras["roc_auc"] = float(roc_auc)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title(f"{split.title()} | ROC Curve")
                ax.legend(loc="lower right")
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                fig.tight_layout()
                roc_path = output_dir / f"{split}_roc_curve.png"
                fig.savefig(roc_path, dpi=200)
                plt.close(fig)
                plots.append(roc_path)
            except ValueError:
                pass

            neg_mask = y_true == 0
            pos_mask = y_true == 1
            if neg_mask.any() and pos_mask.any():
                neg_vals = prob_np[neg_mask]
                pos_vals = prob_np[pos_mask]
                neg_vals = neg_vals[np.isfinite(neg_vals)]
                pos_vals = pos_vals[np.isfinite(pos_vals)]
                if neg_vals.size > 1 and pos_vals.size > 1:
                    neg_range = np.max(neg_vals) - np.min(neg_vals)
                    pos_range = np.max(pos_vals) - np.min(pos_vals)
                    if neg_range > 0 or pos_range > 0:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        ax.hist(neg_vals, bins=min(30, max(2, neg_vals.size)), alpha=0.6, label="Negative", color="tab:blue")
                        ax.hist(pos_vals, bins=min(30, max(2, pos_vals.size)), alpha=0.6, label="Positive", color="tab:orange")
                        ax.set_xlabel("Predicted probability")
                        ax.set_ylabel("Count")
                        ax.set_title(f"{split.title()} | Probability Distribution")
                        ax.legend()
                        fig.tight_layout()
                        hist_path = output_dir / f"{split}_probability_hist.png"
                        fig.savefig(hist_path, dpi=200)
                        plt.close(fig)
                        plots.append(hist_path)
    return plots, extras


def create_multilabel_plots(
    targets: torch.Tensor,
    probabilities: Optional[torch.Tensor],
    split: str,
    output_dir: Path,
    task_info: TaskInfo,
    top_k: int = 5,
) -> List[Path]:
    if probabilities is None or probabilities.numel() == 0 or targets.numel() == 0:
        return []
    y_true = targets.view(-1, task_info.num_classes or probabilities.shape[-1]).numpy()
    y_prob = probabilities.view_as(targets).numpy()
    class_ap = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() == 0:
            continue
        try:
            class_ap.append((c, average_precision_score(y_true[:, c], y_prob[:, c])))
        except Exception:
            continue
    saved: List[Path] = []
    for c, ap in sorted(class_ap, key=lambda x: x[1], reverse=True)[: top_k if top_k > 0 else y_true.shape[1]]:
        try:
            precision, recall, _ = precision_recall_curve(y_true[:, c], y_prob[:, c])
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(recall, precision, label=f"AP={ap:.3f}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"{split.title()} | Class {c} PR Curve")
            ax.legend(loc="upper right")
            fig.tight_layout()
            path = output_dir / f"{split}_class{c}_pr_curve.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            saved.append(path)
        except Exception:
            continue
    return saved


def compute_subset_metrics(
    task_info: TaskInfo,
    preds: torch.Tensor,
    targets: torch.Tensor,
    probabilities: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    if preds.numel() == 0 or targets.numel() == 0:
        return {}
    if task_info.task_type == "regression":
        meter = RegressionMeter()
        meter.update(preds, targets)
        return meter.compute()
    if task_info.task_type == "multi_target_regression":
        meter = MultiTargetRegressionMeter(task_info.target_names)
        meter.update(preds, targets)
        return meter.compute()
    if task_info.task_type == "multilabel":
        meter = MultiLabelMeter(task_info.num_classes or preds.shape[-1])
        meter.update(preds if probabilities is None else probabilities, targets)
        return meter.compute()

    meter = ClassificationMeter(task_info.num_classes or 2)
    meter.update(preds, targets)
    metrics = meter.compute()
    if (task_info.num_classes or 2) <= 2 and probabilities is not None:
        prob_tensor = _select_positive_probs(probabilities)
        if prob_tensor is not None:
            try:
                y_true = targets.view(-1).detach().cpu().numpy()
                prob_np = prob_tensor.view(-1).detach().cpu().numpy()
                fpr, tpr, _ = roc_curve(y_true, prob_np)
                metrics["roc_auc"] = float(auc(fpr, tpr))
            except ValueError:
                pass
    return metrics


def _group_metrics_for_metadata(
    labels: List[Any],
    preds: torch.Tensor,
    targets: torch.Tensor,
    probabilities: Optional[torch.Tensor],
    task_info: TaskInfo,
) -> Dict[str, Dict[str, float]]:
    if not labels or preds.numel() == 0:
        return {}
    label_array = np.array(labels)
    unique_labels = sorted({str(l) for l in label_array if l is not None})
    results: Dict[str, Dict[str, float]] = {}
    for label in unique_labels:
        mask = label_array == label
        if not mask.any():
            continue
        mask_tensor = torch.from_numpy(mask.astype(np.bool_))
        metrics = compute_subset_metrics(
            task_info,
            preds[mask_tensor],
            targets[mask_tensor],
            probabilities[mask_tensor] if probabilities is not None else None,
        )
        if metrics:
            results[label] = metrics
    return results


def _save_json(data: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def save_disaggregated_metrics(
    dataset_name: Optional[str],
    prediction_outputs: Dict[str, Dict[str, Any]],
    task_info: TaskInfo,
    output_dir: Path,
) -> List[Path]:
    if not dataset_name:
        return []
    dataset_name = dataset_name.lower()
    disagg_dir = output_dir / "disaggregated_metrics"
    saved_paths: List[Path] = []

    def _collect_for_field(field: str) -> Dict[str, Dict[str, Dict[str, float]]]:
        field_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for split_name, data in prediction_outputs.items():
            metadata = data.get("metadata", {})
            labels = metadata.get(field)
            preds = data.get("predictions")
            targets = data.get("targets")
            probs = data.get("probabilities")
            if labels is None or preds is None or targets is None:
                continue
            group_metrics = _group_metrics_for_metadata(labels, preds, targets, probs, task_info)
            if group_metrics:
                field_results[split_name] = group_metrics
        return field_results

    if dataset_name == "omol":
        domain_metrics = _collect_for_field("domain")
        if domain_metrics:
            saved_paths.append(_save_json(domain_metrics, disagg_dir / "metrics_omol_domain.json"))
        is_connected_metrics = _collect_for_field("is_connected")
        if is_connected_metrics:
            saved_paths.append(_save_json(is_connected_metrics, disagg_dir / "metrics_omol_is_connected.json"))
    elif dataset_name == "sat":
        hardness_metrics = _collect_for_field("hardness")
        if hardness_metrics:
            saved_paths.append(_save_json(hardness_metrics, disagg_dir / "metrics_sat_hardness.json"))
        category_metrics = _collect_for_field("category")
        if category_metrics:
            saved_paths.append(_save_json(category_metrics, disagg_dir / "metrics_sat_category.json"))

        combo_results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for split_name, data in prediction_outputs.items():
            metadata = data.get("metadata", {})
            hardness = metadata.get("hardness")
            category = metadata.get("category")
            preds = data.get("predictions")
            targets = data.get("targets")
            probs = data.get("probabilities")
            if hardness is None or category is None or preds is None or targets is None:
                continue
            combo_labels = [None if h is None or c is None else f"{c}__{h}" for h, c in zip(hardness, category)]
            group_metrics = _group_metrics_for_metadata(combo_labels, preds, targets, probs, task_info)
            if group_metrics:
                combo_results[split_name] = group_metrics
        if combo_results:
            saved_paths.append(_save_json(combo_results, disagg_dir / "metrics_sat_category_hardness.json"))

    return saved_paths
