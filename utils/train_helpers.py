"""
Helper functions for training script.
Contains utility functions for data loading, path management, result saving,
random seed setting, GPU memory management, model class mapping, and logging.
"""
import argparse
import logging
import os
from typing import Optional
import pickle
import random
import numpy as np
import pandas as pd
import torch

# -----------------------------------------------------------------------------
# Logging: setup_logger, log_experiment_config, log_write (tqdm-safe)
# -----------------------------------------------------------------------------

BANNER_LEN = 60


class _TqdmStreamHandler(logging.StreamHandler):
    """Console handler that uses tqdm.write when available to avoid breaking progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            try:
                from tqdm import tqdm
                tqdm.write(msg)
            except Exception:
                self.stream.write(msg + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(name="train", level=logging.INFO, log_file=None):
    """
    Lightweight logger setup using stdlib logging. Idempotent: repeated calls
    for the same name do not add duplicate handlers.

    Args:
        name: Logger name (e.g. "train").
        level: Log level (default INFO).
        log_file: If set, also append logs to this file.

    Returns:
        logging.Logger configured with console (and optional file) output.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid duplicate handlers: if already configured, return as-is
    if logger.handlers:
        return logger
    fmt = "%(asctime)s | %(levelname)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    console = _TqdmStreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file:
        try:
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except OSError:
            pass
    return logger


def log_experiment_config(
    logger,
    total_experiments,
    model_types,
    hidden_sizes,
    lrs,
    weight_decays,
    dropout_rates,
):
    """
    Log training loop banner and config summary (same content as original print block).
    Banner line length is 60. Uses logger.info only (no print).
    """
    sep = "=" * BANNER_LEN
    logger.info(sep)
    logger.info("Starting training loop: %s experiments", total_experiments)
    logger.info(
        "Models: %s | Hidden sizes: %s",
        model_types,
        hidden_sizes,
    )
    logger.info(
        "Learning rates: %s | Weight decays: %s | Dropout rates: %s",
        lrs,
        weight_decays,
        dropout_rates,
    )
    logger.info(sep)


def log_experiment_start(
    logger,
    experiment_num,
    total_experiments,
    model_type,
    hidden_size,
    lr,
    weight_decay,
    dropout_rate,
):
    """Log per-experiment banner (one block)."""
    sep = "=" * BANNER_LEN
    logger.info(
        "Experiment %s/%s: %s | hidden_size=%s | lr=%s | weight_decay=%s | dropout=%s",
        experiment_num,
        total_experiments,
        model_type.upper(),
        hidden_size,
        lr,
        weight_decay,
        dropout_rate,
    )
    logger.info(sep)


def log_write(logger, msg, level=logging.INFO):
    """
    Log message in a tqdm-safe way. When the console handler is TqdmStreamHandler,
    logger already writes via tqdm.write; use this for explicit log calls from
    inside tqdm loops. Safe when tqdm is not used (handler falls back to stream.write).
    """
    logger.log(level, msg)


def log_dataset_and_batch_info(
    logger,
    train_data,
    val_data,
    batch_size,
    accel_config,
    train_dl,
    num_workers,
    pin_memory,
    use_sector,
    predict_all_chars,
):
    """
    Log dataset and batch information (content lives here so main training code stays minimal).
    """
    logger.info("Dataset Information:")
    logger.info("  Training dataset size: %s samples", len(train_data))
    logger.info("  Validation dataset size: %s samples", len(val_data))
    logger.info("  Batch size per step: %s", batch_size)
    logger.info(
        "  Effective batch size (with grad accum): %s",
        batch_size * accel_config.grad_accum_steps,
    )
    logger.info("  Number of batches per epoch: %s", len(train_dl))
    logger.info("  use_sector: %s, predict_all_chars: %s", use_sector, predict_all_chars)
    logger.info(
        "  acceleration: %s, workers: %s, pin_memory: %s",
        accel_config.use_acceleration,
        num_workers,
        pin_memory and num_workers > 0,
    )
    if num_workers > 0:
        logger.info("  DataLoader prefetch_factor: %s", accel_config.dataloader_prefetch_factor)


def _gpu_has_train_rnn_process(gpu_index: int) -> bool:
    """
    Return True if the given GPU has a Python process running train_rnn_updated (this script).
    Used to avoid placing a second training job on the same GPU.
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        if out.returncode != 0:
            return False
        lines = [x.strip() for x in out.stdout.strip().splitlines() if x.strip()]
        for line in lines:
            if not line.isdigit():
                continue
            pid = int(line)
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmd = f.read().decode("utf-8", errors="ignore").replace("\x00", " ")
                if "train_rnn_updated" in cmd:
                    return True
            except (FileNotFoundError, PermissionError, ValueError):
                continue
        return False
    except Exception:
        return False


def pick_cuda_device_index() -> Optional[int]:
    """
    Choose a CUDA device index: prefer a GPU that has no Python train_rnn_updated task.
    If multiple GPUs exist, returns the first index with no such process; otherwise 0.
    Call this before any torch.cuda init and set CUDA_VISIBLE_DEVICES to the returned index.
    Returns None if no NVIDIA GPU is visible (or nvidia-smi is unavailable).
    """
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        indices = []
        for line in out.stdout.strip().splitlines():
            s = line.strip()
            if not s:
                continue
            try:
                indices.append(int(s))
            except ValueError:
                continue
        if not indices:
            return None
        if len(indices) == 1:
            idx = indices[0]
            # Single GPU: either use it or nothing else is available anyway
            return idx
        # Multi-GPU: prefer one without train_rnn_updated running
        for idx in indices:
            if not _gpu_has_train_rnn_process(idx):
                return idx
        # If all have train_rnn_updated, just fall back to the first
        return indices[0]
    except Exception:
        return None


def save_results(results, filepath):
    """
    Save training results to local file
    
    Args:
        results: Training results dictionary
        filepath: Save path (e.g., 'results_rnn' or 'results/rnn_sector')
                  If directory doesn't exist, it will be created
    """
    # Extract directory and filename
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create save dictionary (does not include model, because model is too large)
    results_path = filepath + '.pkl'
    save_dict = {}
    for key, value in results.items():
        if key != "model":
            save_dict[key] = value
    
    # Save model state dict (can be saved separately if needed)
    model_path = results_path.replace('.pkl', '_model.pth')
    if "model" in results:
        # Check if model has fcpos (position prediction layer)
        # In predict_all_chars mode, fcpos is None
        if hasattr(results["model"], 'fcpos') and results["model"].fcpos is not None:
            saved_num_pos = results["model"].fcpos.out_features
            print(f"Model has position prediction layer (num_pos={saved_num_pos})")
        elif hasattr(results["model"], 'predict_all_chars') and results["model"].predict_all_chars:
            print("Model is in predict_all_chars mode (no position prediction)")
        else:
            print("Model does not have position prediction layer")
        torch.save(results["model"].state_dict(), model_path)
        print(f"Model state dict saved to: {model_path}")
    
    # Save other results
    with open(results_path, 'wb') as f:
        pickle.dump(save_dict, f)
    print(f"Results saved to: {results_path}")


def get_base_path(override: Optional[str] = None) -> str:
    """Get the base directory for stimulus/label files. Adapts to current environment.

    Resolution order:
    1. override: if provided (e.g. from --data_dir), use it after expanding user and resolving to absolute path.
    2. Environment: AIM3_STIMULI_PATH or FAW_RNN_DATA_PATH (first set wins).
    3. Project-relative: <repo_root>/stimuli, where repo_root is the directory containing the 'utils' package.
    """
    if override and override.strip():
        base_path = os.path.abspath(os.path.expanduser(override.strip()))
        print(f"Using base path (override): {base_path}")
        return base_path
    env_path = os.environ.get("AIM3_STIMULI_PATH") or os.environ.get("FAW_RNN_DATA_PATH")
    if env_path and env_path.strip():
        base_path = os.path.abspath(os.path.expanduser(env_path.strip()))
        print(f"Using base path (env): {base_path}")
        return base_path
    # Project-relative: this file is in <repo>/utils/, so repo root is parent of utils
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.dirname(_this_dir)
    base_path = os.path.join(_repo_root, "stimuli")
    print(f"Using base path (project-relative): {base_path}")
    return base_path


def prepare_data_paths(base_path: str, data_suffix: str = "", splits: tuple = ("train", "valid")):
    """Construct and validate stimulus / label file paths.

    Args:
        base_path: Base directory containing stimulus/label files.
        data_suffix: Optional suffix appended to base filenames.
            - Default ""  -> use filenames without any suffix, e.g. 'stimulus_reg-train.npy'
            - Non-empty (e.g. "cplx", "40h") -> a leading hyphen is added automatically,
              resulting in filenames like 'stimulus_reg-train-cplx.npy'.
        splits: Which splits to prepare and validate. One or more of "train", "valid", "test".
            - ("train", "valid") -> returns (stim_train_path, label_train_path, stim_val_path, label_val_path)
            - ("test",) -> returns (stim_test_path, label_test_path)
            - ("train", "valid", "test") -> returns (..., stim_test_path, label_test_path)  (6 elements)

    Returns:
        Tuple of paths for the requested splits only (see splits above).
    """
    # Normalize suffix: ensure a single leading hyphen when non-empty
    if data_suffix:
        if not data_suffix.startswith("-"):
            suffix = f"-{data_suffix}"
        else:
            suffix = data_suffix
    else:
        suffix = ""

    # (file basename without extension, display name for logs)
    _path_spec = {
        "train": ("stimulus_reg-train", "train"),
        "valid": ("stimulus_reg-validation", "val"),
        "test": ("stimulus_reg-test", "test"),
    }
    out = []
    checks = []
    for name in splits:
        if name not in _path_spec:
            raise ValueError(f"Unknown split: {name}. Must be one of {list(_path_spec)}")
        base, label = _path_spec[name]
        stim_path = os.path.join(base_path, f"{base}{suffix}.npy")
        label_path = os.path.join(base_path, f"{base}{suffix}.tsv")
        out.extend([stim_path, label_path])
        checks.extend([(f"{label} stim", stim_path), (f"{label} label", label_path)])

    print("Checking data paths...")
    print(f"Base path: {base_path}")
    for path_name, path in checks:
        if not os.path.exists(path):
            print(f"ERROR: {path_name} path does not exist: {path}")
            raise FileNotFoundError(f"Data file not found: {path}")
        print(f"  ✓ {path_name}: {path}")

    return tuple(out)


def load_raw_data(stim_train_path: str, label_train_path: str,
                  stim_val_path: str, label_val_path: str,
                  use_mmap: bool = False,
                  paths_tuple=None):
    """Load raw numpy and label data from disk.

    Args:
        stim_train_path: Path to training stimuli numpy file (ignored if paths_tuple is set).
        label_train_path: Path to training labels TSV file (ignored if paths_tuple is set).
        stim_val_path: Path to validation stimuli numpy file (ignored if paths_tuple is set).
        label_val_path: Path to validation labels TSV file (ignored if paths_tuple is set).
        use_mmap: If True, load stimuli with mmap_mode='r' (memory-mapped, use num_workers=0).
                  If False, load as ndarray in memory so DataLoader can use num_workers > 0.
        paths_tuple: Optional. If set, must be the tuple returned by prepare_data_paths(...).
            Length 4 -> load train and valid only; length 2 -> load test only; length 6 -> load all.
            When provided, the four path args above are ignored.

    Returns:
        - If paths_tuple has length 2 (test only): (stims_test, lbls_test)
        - If paths_tuple has length 4 or not set (train+valid): (stims_train, lbls_train, stims_val, lbls_val)
        - If paths_tuple has length 6: (stims_train, lbls_train, stims_val, lbls_val, stims_test, lbls_test)
    """
    if paths_tuple is not None:
        n = len(paths_tuple)
        if n == 2:
            stim_path, label_path = paths_tuple
            return _load_single_split(stim_path, label_path, "test", use_mmap)
        if n == 4:
            stim_train_path, label_train_path, stim_val_path, label_val_path = paths_tuple
            return _load_train_val(stim_train_path, label_train_path, stim_val_path, label_val_path, use_mmap)
        if n == 6:
            (stim_train_path, label_train_path, stim_val_path, label_val_path,
             stim_test_path, label_test_path) = paths_tuple
            train_val = _load_train_val(stim_train_path, label_train_path, stim_val_path, label_val_path, use_mmap)
            test_pair = _load_single_split(stim_test_path, label_test_path, "test", use_mmap)
            return train_val + test_pair
        raise ValueError(f"paths_tuple must have length 2, 4, or 6, got {n}")

    return _load_train_val(stim_train_path, label_train_path, stim_val_path, label_val_path, use_mmap)


def _load_train_val(stim_train_path, label_train_path, stim_val_path, label_val_path, use_mmap):
    """Load train and validation arrays only."""
    print("\nLoading data (train, valid)...")
    if use_mmap:
        stims_train = np.load(stim_train_path, allow_pickle=True, mmap_mode="r")
        stims_val = np.load(stim_val_path, allow_pickle=True, mmap_mode="r")
        print(f"  ✓ Loaded training stimuli (mmap): {stims_train.shape}")
        print(f"  ✓ Loaded validation stimuli (mmap): {stims_val.shape}")
    else:
        stims_train = np.load(stim_train_path, allow_pickle=True)
        stims_val = np.load(stim_val_path, allow_pickle=True)
        print(f"  ✓ Loaded training stimuli (ndarray): {stims_train.shape}")
        print(f"  ✓ Loaded validation stimuli (ndarray): {stims_val.shape}")
    lbls_train = pd.read_csv(label_train_path, sep="\t", index_col=0)
    print(f"  ✓ Loaded training labels: {lbls_train.shape}")
    lbls_val = pd.read_csv(label_val_path, sep="\t", index_col=0)
    print(f"  ✓ Loaded validation labels: {lbls_val.shape}")
    return stims_train, lbls_train, stims_val, lbls_val


def _load_single_split(stim_path, label_path, split_name: str, use_mmap: bool):
    """Load a single split (e.g. test) and return (stims, lbls)."""
    print(f"\nLoading data ({split_name})...")
    if use_mmap:
        stims = np.load(stim_path, allow_pickle=True, mmap_mode="r")
        print(f"  ✓ Loaded {split_name} stimuli (mmap): {stims.shape}")
    else:
        stims = np.load(stim_path, allow_pickle=True)
        print(f"  ✓ Loaded {split_name} stimuli (ndarray): {stims.shape}")
    lbls = pd.read_csv(label_path, sep="\t", index_col=0)
    print(f"  ✓ Loaded {split_name} labels: {lbls.shape}")
    return (stims, lbls)


def create_datasets(stims_train, lbls_train, stims_val, lbls_val,
                    use_sector_mode: bool, predict_all_chars: bool,
                    max_chars: int = 10, dataset_class=None,
                    splits: tuple = ("train", "valid"),
                    stims_test=None, lbls_test=None):
    """Create train/validation/test dataset(s) and return dataset objects and num_pos.

    Args:
        stims_train: Training stimuli numpy array (required when "train" in splits).
        lbls_train: Training labels DataFrame (required when "train" in splits).
        stims_val: Validation stimuli numpy array (required when "valid" in splits).
        lbls_val: Validation labels DataFrame (required when "valid" in splits).
        use_sector_mode: Whether to use sector mode (3x3 grid).
        predict_all_chars: Whether to predict all characters (fg+bg).
        max_chars: Maximum number of characters per frame (for predict_all_chars mode).
        dataset_class: Dataset class to use (MC_RNN_Dataset). If None, will raise error.
        splits: Which splits to create. ("train", "valid") | ("test",) | ("train", "valid", "test").
        stims_test: Test stimuli (required when "test" in splits).
        lbls_test: Test labels (required when "test" in splits).

    Returns:
        - splits=("train", "valid"): (train_ds, val_ds, num_pos)
        - splits=("test",): (test_ds, num_pos)
        - splits=("train", "valid", "test"): (train_ds, val_ds, test_ds, num_pos)
    """
    if dataset_class is None:
        raise ValueError("dataset_class must be provided (e.g., MC_RNN_Dataset)")

    print("Creating datasets...")
    if "test" in splits and (stims_test is None or lbls_test is None):
        raise ValueError("stims_test and lbls_test are required when 'test' is in splits")

    if predict_all_chars:
        num_pos = 0
        _kw = {"use_sector": False, "predict_all_chars": True, "max_chars": max_chars}
        print(f"Using all-chars mode: predict all characters (fg+bg) per frame, max_chars={max_chars}")
    elif use_sector_mode:
        num_pos = 9
        _kw = {"use_sector": True, "num_sectors": num_pos, "predict_all_chars": False}
        print("Using sector mode (3x3 grid, 9 sectors)")
    else:
        num_pos = 2
        _kw = {"use_sector": False, "predict_all_chars": False}
        print("Using coordinate mode (directly predict x, y coordinates)")

    def make_ds(stims, lbls):
        return dataset_class(stims, lbls, **_kw)

    # (stims, lbls) per split in fixed order
    _data_by_split = [
        ("train", stims_train, lbls_train),
        ("valid", stims_val, lbls_val),
        ("test", stims_test, lbls_test),
    ]
    out = [make_ds(stims, lbls) for name, stims, lbls in _data_by_split if name in splits]
    out.append(num_pos)
    return tuple(out)


def set_seed(seed: int):
    """Set random seed for reproducibility (Python, NumPy, PyTorch, CUDA when available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id, seed):
    """Worker initialization function for DataLoader workers.
    This function must be at module level to be picklable for multiprocessing.
    
    Args:
        worker_id: Worker process ID
        seed: Base random seed
    """
    np.random.seed(seed + worker_id)
    torch.manual_seed(seed + worker_id)


def get_model_classes(rnn_conv_class, lstm_conv_class, gru_conv_class, 
                      gawf_rnn_conv_class, feedforward_conv_class, dendritic_ann_conv_class):
    """Return mapping from model type name to model class.
    
    Args:
        rnn_conv_class: RNNConv class
        lstm_conv_class: LSTMConv class
        gru_conv_class: GRUConv class
        gawf_rnn_conv_class: GaWFRNNConv class
        feedforward_conv_class: FeedForwardConv class
        dendritic_ann_conv_class: DendriticANNConv class
    
    Returns:
        Dictionary mapping model type names to model classes
    """
    return {
        "rnn": rnn_conv_class,
        "lstm": lstm_conv_class,
        "gru": gru_conv_class,
        "gawf": gawf_rnn_conv_class,
        "ffn": feedforward_conv_class,
        "dann": dendritic_ann_conv_class,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for command line options."""
    parser = argparse.ArgumentParser(description="Train RNN models for sector classification")
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",
        default=["rnn"],
        choices=["rnn", "lstm", "gru", "gawf", "ffn", "dann"],
        help='Model types to train (default: ["rnn"])',
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=[256],
        help="Hidden sizes to test (default: [256])",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=[0.001],
        help="Learning rates to search over (default: [0.001])",
    )
    parser.add_argument(
        "--wds",
        type=float,
        nargs="+",
        default=[0],
        help="Weight decay values to search over (default: [1e-4])",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "muon"],
        help="Optimizer (optim) to use: 'adam', 'adamw', or 'muon' (default: 'adam')",
    )
    parser.add_argument(
        "--dropouts",
        dest="dropouts",
        type=float,
        nargs="+",
        default=[0],
        help="Dropout rates to search over for model creation (default: [0.3])",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--use_acceleration",
        action="store_true",
        default=True,
        help="Enable acceleration features for training (default: False)",
    )
    parser.add_argument(
        "--use_sector_mode",
        action="store_true",
        default=True,
        help="Use sector mode (3x3 grid, 9 sectors) instead of coordinate mode (default: False)",
    )
    parser.add_argument(
        "--use_mmap",
        action="store_true",
        default=True,
        help=(
            "Load stimuli with memory mapping (mmap_mode='r'). "
            "If not set, load as ndarray in memory so num_workers can be used (default: False)"
        ),
    )
    parser.add_argument(
        "--predict_all_chars",
        action="store_true",
        default=False,
        help="Predict all characters (fg+bg) per frame instead of only foreground character (default: False)",
    )
    parser.add_argument(
        "--nofb",
        action="store_true",
        default=False,
        help=(
            "GaWFRNN only: disable feedback. Behavior: "
            "(1) omit --nofb -> full feedback throughout. "
            "(2) use --nofb only -> no feedback throughout. "
            "(3) use --nofb and --fb_start_epoch N -> no feedback until epoch N, then feedback on"
        ),
    )
    parser.add_argument(
        "--fb_start_epoch",
        type=int,
        default=999999,
        help="GaWFRNN with --nofb: 0-based epoch at which to turn on feedback and unfreeze U,V.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help=(
            "Base directory containing stimulus_reg-* files. "
            "If not set, uses AIM3_STIMULI_PATH / FAW_RNN_DATA_PATH env, else <repo>/stimuli."
        ),
    )
    parser.add_argument(
        "--data_suffix",
        type=str,
        default="",
        help=(
            "Suffix appended to stimulus_reg-* file names. "
            "Example: 'cplx' -> 'stimulus_reg-train-cplx.npy'. "
            "Default: empty string (no suffix)."
        ),
    )
    parser.add_argument(
        "--result_suffix",
        type=str,
        default="sector",
        help="Suffix to append to result file names for distinguishing different training runs (default: empty string)",
    )
    return parser

