import os
import sys
import json
import torch

from prepare_gawf import prepare_experiment, network_train
from utils.train_helpers import save_results, log_experiment_config, log_experiment_start


# Minimal config for autoresearch
DATA_SUFFIX = ""           # "" = 4h, "40h" = 40h
DATASET_MODE = "sector"    # "sector" | "coord" | "allchars"
NUM_EPOCHS = 20
SEED = 42
USE_MMAP = True
USE_ACCELERATION = True
RESULT_SUFFIX = "rnn"

MODEL_PHASES = ["rnn"]

HIDDEN_SIZE = 256
LR = 0.001  # Increased learning rate to improve training
WEIGHT_DECAY = 0.0  # Removed weight decay to avoid over-constraining
DROPOUT = 0.3  # Kept dropout at 0.3 to balance overfitting and training accuracy
OPTIMIZER = "adamw"
NOFB = False
FB_START_EPOCH = 999999

PHASE_OVERRIDES = {
    # "gawf": {"lr": 0.0005, "nofb": True, "fb_start_epoch": 10},
}


def _tqdm_ok():
    term_ok = os.environ.get("TERM", "").lower() not in ("", "dumb")
    return not os.environ.get("DISABLE_TQDM", "").lower() in ("1", "true", "yes") and (
        os.environ.get("ENABLE_TQDM", "").lower() in ("1", "true", "yes") or (sys.stdout.isatty() and term_ok)
    )


def run_one_experiment(prep, model_type, hidden_size, lr, weight_decay, dropout_rate, num_epochs, optim, nofb, fb_start_epoch):
    train_ds = prep["train_ds"]
    val_ds = prep["val_ds"]
    num_pos = prep["num_pos"]
    model_classes = prep["model_classes"]
    logger = prep["logger"]
    device = prep["device"]
    max_chars = prep["max_chars"]
    predict_all_chars = prep["predict_all_chars"]
    results_dir = prep["results_dir"]

    if model_type not in model_classes:
        logger.warning("Unsupported model_type %s, skipping", model_type)
        return None

    eff_num_pos = 0 if predict_all_chars else num_pos
    ModelClass = model_classes[model_type]
    mdl = ModelClass(
        num_classes=10,
        num_pos=eff_num_pos,
        kernel_size=5,
        device=device,
        dropout_rate=dropout_rate,
        hidden_size=hidden_size,
        max_chars=max_chars,
        predict_all_chars=predict_all_chars,
    )

    logger.info(
        "Created %s (predict_all_chars=%s, max_chars=%s, dropout=%s, hidden_size=%s)",
        model_type.upper(), predict_all_chars, max_chars, dropout_rate, hidden_size,
    )

    results = network_train(
        mdl,
        train_ds,
        val_ds,
        num_epochs=num_epochs,
        lr=lr,
        use_acceleration=USE_ACCELERATION,
        weight_decay=weight_decay,
        dropout_rate=dropout_rate,
        rnn_diag_lambda=1e-4,
        use_mmap=prep["use_mmap"],
        use_tqdm=_tqdm_ok(),
        nofb=nofb,
        fb_start_epoch=fb_start_epoch,
        seed=SEED,
        logger=logger,
        optim=optim,
    )

    actual_epochs = results.get("actual_epochs", num_epochs)
    val_acc_char = results.get("val_acc_char")
    val_acc_pos = results.get("val_acc_pos")

    best_val_acc_char = float(max(val_acc_char)) if val_acc_char is not None and len(val_acc_char) else None
    final_val_acc_char = float(val_acc_char[-1]) if val_acc_char is not None and len(val_acc_char) else None
    best_val_acc_pos = float(max(val_acc_pos)) if val_acc_pos is not None and len(val_acc_pos) else None
    final_val_acc_pos = float(val_acc_pos[-1]) if val_acc_pos is not None and len(val_acc_pos) else None

    mode_suffix = "allchars" if predict_all_chars else ("sector" if prep["use_sector_mode"] else "coord")
    acc_suffix = "_acc" if USE_ACCELERATION else ""
    hp_suffix = f"_lr{lr}_wd{weight_decay}_do{dropout_rate}"
    fb_path_suffix = "_nofb" if nofb and fb_start_epoch >= 999999 else (f"_fb{fb_start_epoch}" if nofb else "")
    results_path = os.path.join(
        results_dir,
        f"{model_type}_{mode_suffix}{acc_suffix}_h{hidden_size}{hp_suffix}{fb_path_suffix}",
    )
    save_results(results, results_path)

    metric_summary = {
        "model_type": model_type,
        "dataset_suffix": DATA_SUFFIX,
        "dataset_mode": DATASET_MODE,
        "num_epochs": num_epochs,
        "hidden_size": hidden_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout_rate,
        "optimizer": optim,
        "actual_epochs": actual_epochs,
        "best_val_acc_char": best_val_acc_char,
        "best_val_acc_pos": best_val_acc_pos,
        "final_val_acc_char": final_val_acc_char,
        "final_val_acc_pos": final_val_acc_pos,
    }
    metric_path = os.path.join(results_dir, "metric.json")
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(metric_summary, f, indent=2)
    logger.info("Wrote %s", metric_path)

    return metric_summary


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    use_sector_mode = DATASET_MODE == "sector"
    predict_all_chars = DATASET_MODE == "allchars"

    prep = prepare_experiment(
        data_suffix=DATA_SUFFIX,
        use_sector_mode=use_sector_mode,
        predict_all_chars=predict_all_chars,
        use_mmap=USE_MMAP,
        seed=SEED,
        data_dir=None,
        result_suffix=RESULT_SUFFIX,
    )

    logger = prep["logger"]
    model_classes = prep["model_classes"]
    total = len(MODEL_PHASES)
    log_experiment_config(
        logger, total,
        MODEL_PHASES,
        [HIDDEN_SIZE],
        [LR],
        [WEIGHT_DECAY],
        [DROPOUT],
    )

    for idx, model_type in enumerate(MODEL_PHASES):
        if model_type not in model_classes:
            logger.warning("Skipping unknown model_type: %s", model_type)
            continue
        overrides = PHASE_OVERRIDES.get(model_type, {})
        num_epochs = overrides.get("num_epochs", NUM_EPOCHS)
        hidden_size = overrides.get("hidden_size", HIDDEN_SIZE)
        lr = overrides.get("lr", LR)
        weight_decay = overrides.get("weight_decay", WEIGHT_DECAY)
        dropout_rate = overrides.get("dropout", DROPOUT)
        optim = overrides.get("optimizer", OPTIMIZER)
        nofb = overrides.get("nofb", NOFB)
        fb_start_epoch = overrides.get("fb_start_epoch", FB_START_EPOCH)

        log_experiment_start(
            logger, idx + 1, total,
            model_type, hidden_size, lr, weight_decay, dropout_rate,
        )
        run_one_experiment(
            prep,
            model_type,
            hidden_size=hidden_size,
            lr=lr,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            num_epochs=num_epochs,
            optim=optim,
            nofb=nofb,
            fb_start_epoch=fb_start_epoch,
        )

    logger.info("All phases completed. Results in %s", prep["results_dir"])


if __name__ == "__main__":
    main()
