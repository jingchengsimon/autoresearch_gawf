"""
Fixed preparation layer for GAWF training. Not edited by autoresearch.

Provides: MC_RNN_Dataset, prepare_experiment(), network_train().
Data loading, dataset creation, model registry, logger, and seed are handled here.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.train_helpers import (
    get_base_path,
    prepare_data_paths,
    load_raw_data,
    create_datasets,
    set_seed,
    get_model_classes,
    setup_logger,
    pick_cuda_device_index,
)
from engine.train_rnn_engine import (
    setup_training_components,
    cleanup_dataloaders,
    stop_requested,
    prepare_training_epoch,
    train_one_batch,
    finalize_training_epoch,
    run_validation_for_epoch,
    evaluate_epoch,
)
from models.train_rnn_core import RNNConv, GRUConv, LSTMConv
from models.train_gawf_core import GaWFRNNConv
from models.train_ann_core import DendriticANNConv, FeedForwardConv

torch.set_num_threads(4)


# -----------------------------------------------------------------------------
# Dataset (moved from main script)
# -----------------------------------------------------------------------------

class MC_RNN_Dataset(Dataset):
    def __init__(self, data, labels, frame_num=32, chan_num=2, use_sector=False, num_sectors=9,
                 max_chars=15, predict_all_chars=False):
        """
        Args:
            data (np.ndarray): Array of shape (num_frames_total, height, width); supports float32 for faster loading.
            labels (np.ndarray): DataFrame with columns ['fg_char_id', 'fg_char_x', 'fg_char_y', 'bg_char_ids']
            frame_num (int): Number of frames to stack for input as multichannel image
            chan_num (int): Number of channels in the input images. Each channel is a previous frame.
            use_sector (bool): If True, map (x, y) position to sector id 0-(num_sectors-1)
            num_sectors (int): Number of sectors, e.g., 9 means 0-8 sectors (3x3 grid)
            max_chars (int): Maximum number of characters per frame (for padding)
            predict_all_chars (bool): If True, predict all characters (fg+bg), else only fg
        """
        self.data = data
        self.frame_num = frame_num
        self.chan_num = chan_num
        self.use_sector = use_sector
        self.num_sectors = num_sectors
        self.max_chars = max_chars
        self.predict_all_chars = predict_all_chars
        self.data_dtype = getattr(data, "dtype", np.uint8)

        if predict_all_chars:
            self.labels_df = labels
            self.fg_char_ids = labels['fg_char_id'].values
            raw_bg = labels['bg_char_ids'].values
            self.bg_char_ids_parsed = []
            for i in range(len(raw_bg)):
                s = str(raw_bg[i]) if raw_bg[i] is not None else ""
                if s and s != "nan" and s.strip():
                    try:
                        self.bg_char_ids_parsed.append([int(x) for x in s.split(",") if x.strip()])
                    except (ValueError, TypeError):
                        self.bg_char_ids_parsed.append([])
                else:
                    self.bg_char_ids_parsed.append([])
        else:
            self.labels = labels[['fg_char_id', 'fg_char_x', 'fg_char_y']].values
            if use_sector:
                N = self.data.shape[0]
                height = self.data.shape[-2]
                width = self.data.shape[-1]
                grid_size = int(np.sqrt(self.num_sectors))
                if grid_size * grid_size != self.num_sectors:
                    raise ValueError(
                        f"num_sectors={self.num_sectors} is not a perfect square, cannot form grid_size x grid_size grid"
                    )
                x = self.labels[:, 1].astype(np.float64)
                y = self.labels[:, 2].astype(np.float64)
                col = np.clip((x / max(width - 1, 1)) * grid_size, 0, grid_size - 1).astype(np.int64)
                row = np.clip((y / max(height - 1, 1)) * grid_size, 0, grid_size - 1).astype(np.int64)
                sector = row * grid_size + col
                char_id = self.labels[:, 0].astype(np.int64)
                self.labels_sector = np.stack([char_id, sector], axis=1).astype(np.int64, copy=False)
            else:
                self.labels_coord = self.labels.astype(np.float32, copy=False)

    def __len__(self):
        return (self.data.shape[0] - self.chan_num) // self.frame_num

    def __getitem__(self, idx):
        start_idx = (idx * self.frame_num) + self.chan_num
        end_idx = start_idx + self.frame_num
        T, C = self.frame_num, self.chan_num

        block_start = start_idx - (C - 1)
        block_end = end_idx
        block = self.data[block_start:block_end]
        use_strided = (
            block.shape[0] == T + C - 1
            and len(block.shape) == 3
            and block.strides[0] >= 0
            and block.strides[1] >= 0
            and block.strides[2] >= 0
        )
        if use_strided:
            s0, s1, s2 = block.strides[0], block.strides[1], block.strides[2]
            H, W = block.shape[1], block.shape[2]
            stacked_frames = np.lib.stride_tricks.as_strided(
                block, shape=(T, C, H, W), strides=(s0, s0, s1, s2)
            )
        else:
            frames = [self.data[start_idx + i : end_idx + i] for i in range(-(C - 1), 1)]
            stacked_frames = np.stack(frames, axis=1)

        if stacked_frames.dtype != np.float32:
            stacked_frames = stacked_frames.astype(np.float32, copy=False)

        if not stacked_frames.flags.writeable:
            stacked_frames = np.array(stacked_frames, copy=True)

        if self.predict_all_chars:
            all_chars_per_frame = []
            for frame_idx in range(start_idx, end_idx):
                fg_char_id = int(self.fg_char_ids[frame_idx])
                bg_char_ids = self.bg_char_ids_parsed[frame_idx] if frame_idx < len(self.bg_char_ids_parsed) else []
                all_chars = [fg_char_id] + bg_char_ids
                padded_chars = all_chars[: self.max_chars] + [-1] * max(0, self.max_chars - len(all_chars))
                all_chars_per_frame.append(padded_chars)
            labels = np.array(all_chars_per_frame, dtype=np.int64)
        else:
            if self.use_sector:
                labels = self.labels_sector[start_idx:end_idx]
            else:
                labels = self.labels_coord[start_idx:end_idx].copy()

        return stacked_frames, labels


# -----------------------------------------------------------------------------
# Preparation API
# -----------------------------------------------------------------------------

def prepare_experiment(
    data_suffix="",
    use_sector_mode=True,
    predict_all_chars=False,
    use_mmap=True,
    seed=42,
    data_dir=None,
    result_suffix="gawf",
):
    """
    Load data, create datasets, set up logger and device, build model registry.
    Intended to be called once by train_gawf.py. All fixed setup lives here.

    Returns:
        dict with: train_ds, val_ds, num_pos, model_classes, logger, device, max_chars,
                   use_sector_mode, predict_all_chars, use_mmap, results_dir,
                   default_config (num_epochs, etc. for reference).
    """
    set_seed(seed)

    cuda_index = pick_cuda_device_index()
    if cuda_index is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_index)
        device = "cuda"
    else:
        device = "cpu"

    base_path = get_base_path(override=data_dir or None)
    stim_train_path, label_train_path, stim_val_path, label_val_path = prepare_data_paths(
        base_path, data_suffix=data_suffix, splits=("train", "valid")
    )
    stims_train, lbls_train, stims_val, lbls_val = load_raw_data(
        stim_train_path, label_train_path, stim_val_path, label_val_path,
        use_mmap=use_mmap,
    )

    max_chars = 15
    train_ds, val_ds, num_pos = create_datasets(
        stims_train, lbls_train, stims_val, lbls_val,
        use_sector_mode=use_sector_mode,
        predict_all_chars=predict_all_chars,
        max_chars=max_chars,
        dataset_class=MC_RNN_Dataset,
        splits=("train", "valid"),
    )

    model_classes = get_model_classes(
        RNNConv,
        LSTMConv,
        GRUConv,
        GaWFRNNConv,
        FeedForwardConv,
        DendriticANNConv,
    )

    results_dir = os.path.join("results", "models", result_suffix)
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, "train.log")
    logger = setup_logger("train", log_file=log_file)

    default_config = {
        "num_epochs": 50,
        "hidden_size": 256,
        "lr": 0.001,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "optimizer": "adamw",
        "use_acceleration": True,
        "rnn_diag_lambda": 1e-4,
        "nofb": False,
        "fb_start_epoch": 999999,
    }

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "num_pos": num_pos,
        "model_classes": model_classes,
        "logger": logger,
        "device": device,
        "max_chars": max_chars,
        "use_sector_mode": use_sector_mode,
        "predict_all_chars": predict_all_chars,
        "use_mmap": use_mmap,
        "results_dir": results_dir,
        "default_config": default_config,
    }


# -----------------------------------------------------------------------------
# Training skeleton (delegates to engine)
# -----------------------------------------------------------------------------

def network_train(
    mdl,
    train_data,
    val_data,
    num_epochs=50,
    loss_weights=None,
    lr=0.001,
    use_acceleration=False,
    weight_decay=None,
    dropout_rate=None,
    rnn_diag_lambda=1e-4,
    use_mmap=False,
    use_tqdm=True,
    nofb=False,
    fb_start_epoch=999999,
    seed=42,
    logger=None,
    optim="adam",
):
    """
    Train model; supports sector and coordinate mode. Uses engine utilities.
    """
    components = setup_training_components(
        mdl=mdl,
        train_data=train_data,
        val_data=val_data,
        num_epochs=num_epochs,
        loss_weights=loss_weights,
        lr=lr,
        use_acceleration=use_acceleration,
        weight_decay=weight_decay,
        rnn_diag_lambda=rnn_diag_lambda,
        use_mmap=use_mmap,
        use_tqdm=use_tqdm,
        seed=seed,
        logger=logger,
        optim=optim,
    )

    val_every = 1
    epoch = 0
    try:
        for epoch in range(num_epochs):
            mdl.train()
            epoch_ctx = prepare_training_epoch(
                mdl=mdl,
                components=components,
                epoch=epoch,
                num_epochs=num_epochs,
                nofb=nofb,
                fb_start_epoch=fb_start_epoch,
            )
            train_pbar = epoch_ctx["train_pbar"]

            for batch_idx, batch in train_pbar:
                if stop_requested(components["stop_flag"]):
                    raise KeyboardInterrupt
                train_one_batch(epoch_ctx, batch_idx, batch)

            train_online_summary = finalize_training_epoch(epoch_ctx)
            train_eval_dl = components["train_eval_dl"]
            device = components["device"]
            use_tqdm = components["use_tqdm"]
            logger_local = components["logger"]

            train_eval_res = evaluate_epoch(
                mdl=mdl,
                data_loader=train_eval_dl,
                device=device,
                use_tqdm=use_tqdm,
                logger=logger_local,
                use_feedback=epoch_ctx["use_feedback_this_epoch"],
                desc="Train(eval-mode)",
            )

            train_acc_char_arr = components["train_acc_char"]
            train_metric_pos_arr = components["train_metric_pos"]
            train_loss_pos_arr = components["train_loss_pos"]
            train_loss_char_arr = components["train_loss_char"]

            train_acc_char_arr[epoch], train_metric_pos_arr[epoch] = train_eval_res[0], train_eval_res[1]
            if len(train_eval_res) >= 3 and train_eval_res[2] is not None and train_loss_pos_arr is not None:
                train_loss_pos_arr[epoch] = train_eval_res[2]
            if len(train_eval_res) >= 4 and train_eval_res[3] is not None:
                train_loss_char_arr[epoch] = train_eval_res[3]

            metrics_mode = components["metrics_mode"]
            train_eval_str = metrics_mode.format_train_str(
                epoch,
                num_epochs,
                train_acc_char_arr[epoch],
                train_metric_pos_arr[epoch],
            )

            val_str = run_validation_for_epoch(
                mdl=mdl,
                components=components,
                epoch=epoch,
                use_feedback_this_epoch=epoch_ctx["use_feedback_this_epoch"],
                val_every=val_every,
            )
            if logger is not None:
                msg = train_eval_str + val_str
                if train_online_summary:
                    msg += f" | Train(online): {train_online_summary}"
                logger.info(msg)

    except (KeyboardInterrupt, SystemExit):
        if logger is not None:
            logger.info("Training interrupted, cleaning up resources...")
    finally:
        cleanup_dataloaders(components)

    if components["device"] == "cuda" or (
        isinstance(components["device"], str) and components["device"].startswith("cuda:")
    ):
        torch.cuda.empty_cache()

    actual_epochs = epoch + 1
    train_acc_char = components["train_acc_char"]
    val_acc_char = components["val_acc_char"]
    metrics_mode = components["metrics_mode"]
    train_metric_pos = components["train_metric_pos"]
    val_metric_pos = components["val_metric_pos"]
    train_loss_pos = components["train_loss_pos"]
    val_loss_pos = components["val_loss_pos"]
    train_loss_char = components["train_loss_char"]
    val_loss_char = components["val_loss_char"]

    base = {
        "train_acc_char": train_acc_char[:actual_epochs],
        "val_acc_char": val_acc_char[:actual_epochs],
        "model": mdl.to("cpu"),
        "actual_epochs": actual_epochs,
    }
    return metrics_mode.add_pos_to_result_dict(
        base,
        train_metric_pos,
        val_metric_pos,
        actual_epochs,
        train_loss_pos=train_loss_pos,
        val_loss_pos=val_loss_pos,
        train_loss_char=train_loss_char,
        val_loss_char=val_loss_char,
    )
