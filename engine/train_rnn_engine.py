"""
Training engine utilities for RNN/ANN/GaWF models.

This module encapsulates the detailed implementation of:
- loss function construction
- optimizer construction
- acceleration and DataLoader setup
- metrics array initialization

The goal is to keep the main training script's `network_train` function
as a clear skeleton while the heavy logic lives here.
"""

from typing import Any, Dict, Tuple

import gc
import signal
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from .train_acceleration import (
    AccelerationConfig,
    setup_acceleration,
    build_loaders,
    run_forward_with_feedback,
    TrainStepper,
)
from utils.train_helpers import log_dataset_and_batch_info
from tasks.train_predict_all_chars import build_loss_fn_all_chars, AllCharsMetricsMode
from tasks.train_sector import build_loss_fn_single, SingleCharMetricsMode
from models.train_gawf_core import GaWFRNNConv



def get_loss_weights(predict_all_chars, use_sector):
    """Default loss weights: [char_weight, pos_weight]."""
    if predict_all_chars:
        return [1, 0]
    if use_sector:
        return [1, 1]
    return [1, 0.001]


def get_criterion_pos(use_sector):
    """Position criterion: CrossEntropyLoss for sector, MSELoss for coordinate."""
    if use_sector:
        return nn.CrossEntropyLoss()
    return nn.MSELoss()


def create_metrics_mode(
    predict_all_chars: bool,
    use_sector: bool,
    max_chars: int | None,
    device: Any,
):
    """
    Factory for metrics mode, shared by training and evaluation.
    """
    if predict_all_chars:
        return AllCharsMetricsMode(max_chars, device)
    elif use_sector:
        return SingleCharMetricsMode(use_sector)
    else:
        msg = (
            "No metrics mode created: unsupported configuration "
            "(predict_all_chars=False, use_sector=False)."
        )
        print(msg)
        raise RuntimeError(msg)


def setup_training_components(
    mdl: nn.Module,
    train_data,
    val_data,
    num_epochs: int,
    loss_weights,
    lr: float,
    use_acceleration: bool,
    weight_decay: float | None,
    rnn_diag_lambda: float,
    use_mmap: bool,
    use_tqdm: bool,
    seed: int,
    logger,
    optim: str = "adam",
) -> Dict[str, Any]:
    """
    Build all training components:
    - device and acceleration config
    - DataLoaders
    - optimizer and loss function
    - metrics mode and stepper
    - metrics storage arrays

    Returns a dict used by the high-level training loop in `network_train`.
    """
    use_sector = train_data.use_sector
    predict_all_chars = train_data.predict_all_chars
    max_chars = train_data.max_chars if predict_all_chars else None

    if loss_weights is None:
        loss_weights = get_loss_weights(predict_all_chars, use_sector)

    device = mdl.device
    mdl.to(device)

    accel_config = AccelerationConfig(use_acceleration=use_acceleration)

    metrics_mode = create_metrics_mode(predict_all_chars, use_sector, max_chars, device)

    autocast_fn, scaler, batch_size, num_workers, pin_memory = setup_acceleration(
        accel_config,
        device,
    )

    is_gawf = isinstance(mdl, GaWFRNNConv)
    train_dl, train_eval_dl, val_dl = build_loaders(
        train_data,
        val_data,
        batch_size,
        num_workers,
        pin_memory,
        accel_config,
        seed,
    )
    base_batch_size_for_logging = batch_size

    # Optimizer: for GaWF, disable weight decay on U and V; base params use passed wd.
    wd = weight_decay if weight_decay is not None else 0.0
    has_big_hidden = getattr(mdl, "hidden_size", 0) >= 512

    optim = (optim or "adam").lower()
    if optim == "adamw":
        OptimClass = torch.optim.AdamW
    elif optim == "muon":
        if hasattr(torch.optim, "Muon"):
            OptimClass = torch.optim.Muon
        else:
            raise RuntimeError(
                "Selected optimizer 'muon' but torch.optim.Muon is not available in this "
                "PyTorch version. Please upgrade PyTorch or choose a different optimizer."
            )
    else:
        OptimClass = torch.optim.Adam

    # torch.optim.Muon only supports 2D parameters. Our models commonly include Conv2d
    # (4D kernels) and bias vectors (1D), so proactively fall back to a safe optimizer.
    if OptimClass is getattr(torch.optim, "Muon", None):
        non_2d = []
        for name, p in mdl.named_parameters():
            if p is None:
                continue
            if p.dim() != 2:
                non_2d.append((name, tuple(p.size())))
                if len(non_2d) >= 5:
                    break
        if non_2d:
            if logger is not None:
                shown = ", ".join([f"{n}={s}" for n, s in non_2d])
                logger.warning(
                    "Requested optimizer 'muon' but found non-2D parameters (Muon only supports 2D). "
                    "Falling back to AdamW. Examples: %s",
                    shown,
                )
            OptimClass = torch.optim.AdamW

    if is_gawf:
        gawf_params = []
        base_params = []
        for name, param in mdl.named_parameters():
            if "U" in name or "V" in name:
                gawf_params.append(param)
            else:
                base_params.append(param)
        param_groups = [
            {"params": base_params, "weight_decay": wd},
            {"params": gawf_params, "weight_decay": 0.0},
        ]
        optim_kwargs = {"lr": lr}
        if OptimClass in (torch.optim.Adam, torch.optim.AdamW) and has_big_hidden:
            optim_kwargs["eps"] = 1e-6
        optim = OptimClass(param_groups, **optim_kwargs)
    else:
        optim_kwargs = {"lr": lr, "weight_decay": wd}
        if OptimClass in (torch.optim.Adam, torch.optim.AdamW) and has_big_hidden:
            optim_kwargs["eps"] = 1e-6
        optim = OptimClass(mdl.parameters(), **optim_kwargs)
    lr_base = lr

    criterion_char = nn.CrossEntropyLoss()
    criterion_pos = get_criterion_pos(use_sector) if not predict_all_chars else None

    if predict_all_chars:
        loss_fn = build_loss_fn_all_chars(
            mdl,
            criterion_char,
            max_chars,
            device,
            loss_weights,
            rnn_diag_lambda,
        )
    elif use_sector:
        loss_fn = build_loss_fn_single(
            mdl,
            criterion_char,
            criterion_pos,
            loss_weights,
            rnn_diag_lambda,
            device,
        )
    else:
        msg = (
            "No loss function built: unsupported configuration "
            "(predict_all_chars=False, use_sector=False)."
        )
        print(msg)
        raise RuntimeError(msg)

    if logger is not None:
        log_dataset_and_batch_info(
            logger,
            train_data,
            val_data,
            base_batch_size_for_logging,
            accel_config,
            train_dl,
            num_workers,
            pin_memory,
            use_sector,
            predict_all_chars,
        )

    stepper = TrainStepper(
        mdl,
        optim,
        loss_fn,
        accel_config,
        device,
        scaler,
        autocast_fn,
        pin_memory,
    )

    # Metrics storage (character acc, position metric, and optional losses).
    train_acc_char = np.zeros(num_epochs)
    val_acc_char = np.zeros(num_epochs)
    train_metric_pos = np.zeros(num_epochs)
    val_metric_pos = np.zeros(num_epochs)
    train_loss_char = np.zeros(num_epochs)
    val_loss_char = np.zeros(num_epochs)
    train_loss_pos = np.zeros(num_epochs) if use_sector else None
    val_loss_pos = np.zeros(num_epochs) if use_sector else None

    # Stop flag & signal handlers for graceful shutdown
    stop_flag = init_stop_flag()
    register_stop_handlers(num_workers=num_workers, stop_flag=stop_flag, logger=logger)

    return {
        "device": device,
        "accel_config": accel_config,
        "metrics_mode": metrics_mode,
        "autocast_fn": autocast_fn,
        "scaler": scaler,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "is_gawf": is_gawf,
        "train_dl": train_dl,
        "train_eval_dl": train_eval_dl,
        "val_dl": val_dl,
        "base_batch_size_for_logging": base_batch_size_for_logging,
        "optim": optim,
        "lr_base": lr_base,
        "criterion_char": criterion_char,
        "criterion_pos": criterion_pos,
        "loss_fn": loss_fn,
        "stepper": stepper,
        "train_acc_char": train_acc_char,
        "val_acc_char": val_acc_char,
        "train_metric_pos": train_metric_pos,
        "val_metric_pos": val_metric_pos,
        "train_loss_char": train_loss_char,
        "val_loss_char": val_loss_char,
        "train_loss_pos": train_loss_pos,
        "val_loss_pos": val_loss_pos,
        "use_sector": use_sector,
        "predict_all_chars": predict_all_chars,
        "max_chars": max_chars,
        "logger": logger,
        "use_tqdm": use_tqdm,
        "stop_flag": stop_flag,
    }


def evaluate_epoch(
    mdl: nn.Module,
    data_loader,
    device,
    use_tqdm: bool,
    logger,
    use_feedback,
    desc: str = "Validation",
) -> Tuple:
    """
    Run a full validation epoch and return metrics:
    (acc_char, metric_pos) or (acc_char, metric_pos, loss_pos, loss_char).
    """
    ds = data_loader.dataset
    # Some loaders (e.g., train_eval_dl) may wrap the real dataset in a Subset.
    # Read flags from the underlying dataset to keep metrics mode consistent.
    base_ds = getattr(ds, "dataset", ds)
    predict_all_chars = getattr(base_ds, "predict_all_chars", False)
    use_sector = getattr(base_ds, "use_sector", False)
    max_chars = getattr(base_ds, "max_chars", 10)
    eval_mode = create_metrics_mode(
        predict_all_chars,
        use_sector,
        max_chars if predict_all_chars else None,
        device,
    )
    acc = eval_mode.init_eval()

    val_desc = desc
    if logger is not None:
        logger.info(val_desc)
    valid_pbar = tqdm(
        enumerate(data_loader),
        total=len(data_loader),
        desc=val_desc,
        ncols=100,
        leave=False,
        disable=not use_tqdm,
    )

    mdl.eval()
    with torch.no_grad():
        for _, batch in valid_pbar:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, labels = batch[0], batch[1]
            else:
                inputs, labels = batch, None

            if inputs.dtype == torch.float64:
                inputs = inputs.float()
            inputs = inputs.to(device)
            if labels is not None:
                if labels.dtype == torch.float64:
                    labels = labels.float()
                labels = labels.to(device)

            out_char, out_pos = run_forward_with_feedback(
                mdl,
                inputs,
                use_feedback=use_feedback,
            )
            # Single-char and all-chars modes have different eval update signatures.
            if isinstance(eval_mode, AllCharsMetricsMode):
                acc = eval_mode.update_eval_batch(acc, out_char, labels)
            else:
                acc = eval_mode.update_eval_batch(acc, out_char, labels, out_pos)

    result = eval_mode.finalize_eval(acc, len(data_loader))
    # Normalize to (acc_char, metric_pos, loss_pos, loss_char) where loss_* may be None.
    if len(result) == 2:
        acc_char, metric_pos = result
        loss_pos, loss_char = None, None
        return acc_char, metric_pos, loss_pos, loss_char
    return result


def cleanup_dataloaders(state: Dict[str, Any]) -> None:
    """
    Explicit resource cleanup for DataLoader so worker processes exit.
    Designed to be called from the main training script's `finally` block.
    """
    num_workers = state.get("num_workers", 0)
    if num_workers <= 0:
        return

    try:
        logger = state.get("logger", None)
        train_dl = state.get("train_dl")
        train_eval_dl = state.get("train_eval_dl")
        val_dl = state.get("val_dl")

        if train_dl is not None:
            train_dl._iterator = None
            del train_dl
        if train_eval_dl is not None:
            train_eval_dl._iterator = None
            del train_eval_dl
        if val_dl is not None:
            val_dl._iterator = None
            del val_dl

        gc.collect()
        if logger is not None:
            logger.info("DataLoader workers cleaned up successfully")
    except Exception as e:
        logger = state.get("logger", None)
        if logger is not None:
            logger.warning("Error during DataLoader cleanup: %s", e)


def init_stop_flag() -> Dict[str, bool]:
    """Create a mutable stop flag used for graceful shutdown."""
    return {"requested": False}


def register_stop_handlers(num_workers: int, stop_flag: Dict[str, bool], logger=None) -> None:
    """
    Register SIGTERM/SIGINT handlers that set a stop flag.
    DataLoader workers are not touched inside the handler.
    """
    if num_workers <= 0:
        return

    def _request_stop(signum, frame):
        stop_flag["requested"] = True
        if logger is not None:
            logger.info("[signal] got %s, will stop after current step...", signum)

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)


def stop_requested(stop_flag: Dict[str, bool]) -> bool:
    """Check whether a stop was requested by signal handlers."""
    return bool(stop_flag.get("requested", False))


def use_feedback_this_epoch(epoch: int, is_gawf: bool, nofb: bool, fb_start_epoch: int):
    """
    GaWFRNN: single nofb-based control for feedback.
    This governs whether the feedback path is used and when U,V are frozen.
    """
    if not is_gawf:
        return None  # not used
    if not nofb:
        return True  # default: always feedback
    return epoch >= fb_start_epoch


def prepare_training_epoch(
    mdl: nn.Module,
    components: Dict[str, Any],
    epoch: int,
    num_epochs: int,
    nofb: bool,
    fb_start_epoch: int,
) -> Dict[str, Any]:
    """
    Prepare one training epoch:
    - optionally freeze/unfreeze feedback parameters (GaWF nofb)
    - init epoch metrics and tqdm progress bar
    """
    is_gawf = components["is_gawf"]
    train_dl = components["train_dl"]
    metrics_mode = components["metrics_mode"]
    use_tqdm = components["use_tqdm"]
    logger = components["logger"]

    feedback_flag = use_feedback_this_epoch(epoch, is_gawf, nofb, fb_start_epoch)

    if is_gawf and nofb:
        if hasattr(mdl, "set_feedback_frozen"):
            mdl.set_feedback_frozen(feedback_flag is False)
        if use_tqdm and (epoch == 0 or epoch == fb_start_epoch) and logger is not None:
            logger.info("GaWFRNN (nofb): epoch %d use_feedback=%s", epoch, feedback_flag)

    epoch_acc = metrics_mode.init_epoch_train()
    num_batches_total = len(train_dl)

    train_desc = f"Epoch {epoch + 1}/{num_epochs} [Train]"
    if logger is not None:
        logger.info(train_desc)
    train_pbar = tqdm(
        enumerate(train_dl),
        total=num_batches_total,
        desc=train_desc,
        ncols=100,
        leave=False,
        disable=not use_tqdm,
    )

    return {
        "epoch": epoch,
        "num_epochs": num_epochs,
        "use_feedback_this_epoch": feedback_flag,
        "metrics_mode": metrics_mode,
        "epoch_acc": epoch_acc,
        "num_batches_total": num_batches_total,
        "train_pbar": train_pbar,
        "components": components,
    }


def train_one_batch(epoch_ctx: Dict[str, Any], batch_idx: int, batch):
    """
    Run one training batch:
    - forward
    - loss computation
    - backward + optimizer step (inside TrainStepper)
    - update training metrics and tqdm postfix
    """
    components = epoch_ctx["components"]
    stepper = components["stepper"]
    metrics_mode = epoch_ctx["metrics_mode"]
    train_dl = components["train_dl"]
    use_feedback_this_epoch = epoch_ctx["use_feedback_this_epoch"]
    use_tqdm = components["use_tqdm"]
    train_pbar = epoch_ctx["train_pbar"]

    current_loss, out_char, out_pos, labels_device = stepper.step(
        batch,
        batch_idx,
        use_feedback_this_epoch,
    )

    epoch_ctx["epoch_acc"] = metrics_mode.update_train_batch(
        epoch_ctx["epoch_acc"],
        out_char,
        labels_device,
        batch_idx,
        len(train_dl),
        out_pos,
    )

    if use_tqdm and batch_idx % 10 == 0:
        train_pbar.set_postfix(
            metrics_mode.postfix_for_pbar(
                current_loss,
                out_char,
                out_pos,
                labels_device,
            )
        )

    return current_loss, out_char, out_pos, labels_device


def finalize_training_epoch(epoch_ctx: Dict[str, Any]) -> str:
    """
    Finalize online training metrics for one epoch and return a concise summary string.
    This uses train-mode outputs accumulated during the epoch and is intended only
    for monitoring; epoch-level arrays are filled from eval-mode metrics elsewhere.
    """
    components = epoch_ctx["components"]
    metrics_mode = epoch_ctx["metrics_mode"]
    epoch_acc = epoch_ctx["epoch_acc"]
    num_batches_total = epoch_ctx["num_batches_total"]
    epoch = epoch_ctx["epoch"]
    num_epochs = epoch_ctx["num_epochs"]

    train_result = metrics_mode.finalize_train_epoch(epoch_acc, num_batches_total)

    acc_char = train_result[0]
    metric_pos = train_result[1]
    loss_pos = train_result[2] if len(train_result) >= 3 else None
    loss_char = train_result[3] if len(train_result) >= 4 else None

    parts = [f"acc_char={acc_char:.2f}", f"metric_pos={metric_pos:.4f}"]
    if loss_pos is not None:
        parts.append(f"loss_pos={loss_pos:.4f}")
    if loss_char is not None:
        parts.append(f"loss_char={loss_char:.4f}")
    return ", ".join(parts)


def run_validation_for_epoch(
    mdl: nn.Module,
    components: Dict[str, Any],
    epoch: int,
    use_feedback_this_epoch,
    val_every: int = 1,
) -> str:
    """
    Optionally run validation for this epoch and return formatted validation string.
    """
    metrics_mode = components["metrics_mode"]
    val_acc_char = components["val_acc_char"]
    val_metric_pos = components["val_metric_pos"]
    val_loss_pos = components["val_loss_pos"]
    val_loss_char = components["val_loss_char"]
    val_dl = components["val_dl"]
    device = components["device"]
    use_tqdm = components["use_tqdm"]
    logger = components["logger"]

    if epoch % val_every == 0:
        # Full validation pass in eval mode (internally uses no_grad()).
        val_res = evaluate_epoch(
            mdl=mdl,
            data_loader=val_dl,
            device=device,
            use_tqdm=use_tqdm,
            logger=logger,
            use_feedback=use_feedback_this_epoch,
            desc="Validation",
        )
        val_acc_char[epoch], val_metric_pos[epoch] = val_res[0], val_res[1]
        if len(val_res) >= 3 and val_res[2] is not None and val_loss_pos is not None:
            val_loss_pos[epoch] = val_res[2]
        if len(val_res) >= 4 and val_res[3] is not None:
            val_loss_char[epoch] = val_res[3]
    else:
        if epoch > 0:
            val_acc_char[epoch] = val_acc_char[epoch - 1]
            val_metric_pos[epoch] = val_metric_pos[epoch - 1]
            if val_loss_pos is not None:
                val_loss_pos[epoch] = val_loss_pos[epoch - 1]
            if val_loss_char is not None:
                val_loss_char[epoch] = val_loss_char[epoch - 1]

    val_str = metrics_mode.format_val_str(
        val_acc_char[epoch],
        val_metric_pos[epoch],
    )
    return val_str
