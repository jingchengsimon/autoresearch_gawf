import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


# =========================
# User config
# =========================

AIDER_MODEL = "openai/QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ"
TRAIN_CMD = "python train_gawf.py"
PROGRAM_FILE = "program_gawf.md"
TRAIN_FILE = "train_gawf.py"
METRICS_FILE = "results/rnn/models/metrics.json"
OUTPUT_LOG = "output.log"
RESULTS_TSV = "results.tsv"
STOP_FILE = "STOP_AUTORESEARCH"
LOG_DIR = "logs"
LOOP_LOG = os.path.join(LOG_DIR, "loop.log")

# Search control
MAX_TOTAL_EXPERIMENTS = 5
MAX_4H_EXPERIMENTS = min(MAX_TOTAL_EXPERIMENTS, 6)
MAX_NO_IMPROVEMENT = 5

# Success criteria, aligned with program.md
VAL_CHAR_TOL = 0.5       # val_acc_char tolerance
TRAIN_CHAR_DROP_LIMIT = 1.0

# If current dataset is 4h and 10 distinct experiments still overfit, switch to 40h
SWITCH_TO_40H_AFTER_4H_TRIALS = 10

# Training timeout in seconds
TRAIN_TIMEOUT_4H = 3 * 3600
TRAIN_TIMEOUT_40H = 8 * 3600

# Aider env
ENV_OVERRIDES = {
    "OPENAI_API_BASE": "http://localhost:8000/v1",
    "OPENAI_API_KEY": "dummy",
}

# If true, auto-commit the metrics/results.tsv snapshot after each run
COMMIT_RESULTS_SNAPSHOT = False


# =========================
# Utilities
# =========================


def ensure_logs_dir() -> None:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)


def log_loop(msg: str) -> None:
    """
    Log a loop status message to both stdout and logs/loop.log.
    """
    text = msg.rstrip("\n")
    print(text)
    try:
        ensure_logs_dir()
        with open(LOOP_LOG, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        # Logging failures should not crash the loop
        pass


def run_cmd(
    cmd: str,
    *,
    timeout: Optional[int] = None,
    capture_output: bool = True,
    check: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        cmd,
        shell=True,
        text=True,
        capture_output=capture_output,
        timeout=timeout,
        check=check,
        env=merged_env,
    )


def file_exists(path: str) -> bool:
    return Path(path).exists()


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_tsv_row(path: str, row: list[str]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write("\t".join(row) + "\n")


def ensure_results_tsv() -> None:
    if not file_exists(RESULTS_TSV):
        append_tsv_row(
            RESULTS_TSV,
            [
                "timestamp",
                "commit",
                "dataset_suffix",
                "model_type",
                "best_val_acc_char",
                "best_val_acc_pos",
                "final_train_acc_char",
                "final_val_acc_char",
                "gap_char",
                "overfit_flag",
                "status",
                "description",
            ],
        )


def get_head_commit() -> str:
    p = run_cmd("git rev-parse --short HEAD", check=True)
    return p.stdout.strip()


def get_head_branch() -> str:
    p = run_cmd("git branch --show-current", check=True)
    return p.stdout.strip()


def git_commit_all(message: str) -> None:
    run_cmd("git add train_gawf.py", check=False)
    p = run_cmd(f'git commit -m {shlex.quote(message)}', capture_output=True)
    if p.returncode != 0:
        # No changes is okay
        if "nothing to commit" not in (p.stdout + p.stderr).lower():
            print(p.stdout)
            print(p.stderr, file=sys.stderr)


def maybe_commit_results_snapshot() -> None:
    if not COMMIT_RESULTS_SNAPSHOT:
        return
    run_cmd(f"git add {shlex.quote(METRICS_FILE)} {shlex.quote(RESULTS_TSV)}", check=False)
    p = run_cmd('git commit -m "chore: snapshot metrics/results"', capture_output=True)
    if p.returncode != 0:
        if "nothing to commit" not in (p.stdout + p.stderr).lower():
            print(p.stdout)
            print(p.stderr, file=sys.stderr)


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def set_dataset_suffix_in_train_file(target_suffix: str) -> bool:
    """
    Minimal, explicit text replacement:
    DATA_SUFFIX = ""
    or
    DATA_SUFFIX = "40h"
    """
    path = Path(TRAIN_FILE)
    text = path.read_text(encoding="utf-8")
    old1 = 'DATA_SUFFIX = ""'
    old2 = 'DATA_SUFFIX = "40h"'
    new = f'DATA_SUFFIX = "{target_suffix}"'
    if old1 in text:
        text = text.replace(old1, new, 1)
        path.write_text(text, encoding="utf-8")
        return True
    if old2 in text:
        text = text.replace(old2, new, 1)
        path.write_text(text, encoding="utf-8")
        return True
    return False


def summarize_metrics(metrics: Dict[str, Any]) -> str:
    keys = [
        "model_type",
        "dataset_suffix",
        "dataset_mode",
        "num_epochs",
        "hidden_size",
        "lr",
        "weight_decay",
        "dropout",
        "optimizer",
        "best_train_acc_char",
        "final_train_acc_char",
        "best_val_acc_char",
        "final_val_acc_char",
        "best_val_acc_pos",
        "final_val_acc_pos",
        "gap_char",
        "gap_pos",
        "overfit_flag",
        "best_epoch_char",
        "best_epoch_pos",
    ]
    lines = ["Experiment finished."]
    for k in keys:
        if k in metrics:
            lines.append(f"{k}: {metrics[k]}")
    lines.extend(
        [
            "",
            "Follow program.md.",
            "Propose the next experiment.",
            "Rules:",
            "- modify ONLY train_gawf.py",
            "- make ONE small change",
            "- prioritize improving val_acc_char",
            "- do not significantly hurt train_acc_char",
            "- if val_acc_char is similar, prefer reducing gap_char",
            "- do NOT restate files",
            "- do NOT explain program.md",
            "- apply the change directly to train_gawf.py",
        ]
    )
    return "\n".join(lines)


def call_aider(prompt: str, trial_index: int) -> None:
    """
    Single-shot aider call. It will edit train_gawf.py and exit.
    Full stdout/stderr are saved to logs/aider_trial_XXX.log.
    """
    ensure_logs_dir()
    log_path = Path(LOG_DIR) / f"aider_trial_{trial_index:03d}.log"

    cmd = (
        f"aider "
        f"--model {shlex.quote(AIDER_MODEL)} "
        f"--map-tokens 0 "
        f"--no-show-model-warnings "
        f"{shlex.quote(PROGRAM_FILE)} "
        f"{shlex.quote(TRAIN_FILE)} "
        f"--message {shlex.quote(prompt)}"
    )

    merged_env = os.environ.copy()
    merged_env.update(ENV_OVERRIDES)

    with open(log_path, "w", encoding="utf-8") as log_f:
        p = subprocess.run(
            cmd,
            shell=True,
            text=True,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=merged_env,
            stdin=subprocess.DEVNULL,
        )

    if p.returncode != 0:
        raise RuntimeError(
            f"Aider call failed for trial {trial_index:03d}, "
            f"see {log_path}"
        )


def run_training(dataset_suffix: str) -> tuple[str, int]:
    timeout = TRAIN_TIMEOUT_40H if dataset_suffix == "40h" else TRAIN_TIMEOUT_4H
    # Stream training logs to both terminal and output.log for easier monitoring
    cmd = f"{TRAIN_CMD} 2>&1 | tee {shlex.quote(OUTPUT_LOG)}"
    try:
        # Let stdout/stderr pass through so the user can see training progress live
        p = run_cmd(cmd, timeout=timeout, capture_output=False)
        return ("ok" if p.returncode == 0 else "train_error", p.returncode)
    except subprocess.TimeoutExpired:
        return ("timeout", 124)


def load_metrics_or_none() -> Optional[Dict[str, Any]]:
    if not file_exists(METRICS_FILE):
        return None
    try:
        return read_json(METRICS_FILE)
    except Exception:
        return None


def is_improved(
    current: Dict[str, Any],
    best: Optional[Dict[str, Any]],
) -> bool:
    """
    Success priority:
    1. higher val_acc_char
    2. if similar val_acc_char (within tolerance), smaller gap_char
    3. must not hurt train_acc_char too much
    """
    if best is None:
        return True

    cur_val = float(current.get("best_val_acc_char", -1e9))
    best_val = float(best.get("best_val_acc_char", -1e9))
    cur_gap = float(current.get("gap_char", 1e9))
    best_gap = float(best.get("gap_char", 1e9))
    cur_train = float(current.get("final_train_acc_char", -1e9))
    best_train = float(best.get("final_train_acc_char", -1e9))

    # Hard reject if train accuracy collapses
    if cur_train < best_train - TRAIN_CHAR_DROP_LIMIT:
        return False

    # Clear val improvement
    if cur_val > best_val + VAL_CHAR_TOL:
        return True

    # Similar val -> prefer lower gap
    if abs(cur_val - best_val) <= VAL_CHAR_TOL and cur_gap < best_gap:
        return True

    return False


def should_switch_to_40h(
    four_h_trials: int,
    last_metrics: Optional[Dict[str, Any]],
    best_metrics_4h: Optional[Dict[str, Any]],
) -> bool:
    if four_h_trials < SWITCH_TO_40H_AFTER_4H_TRIALS:
        return False
    if last_metrics is None:
        return False
    overfit = bool(last_metrics.get("overfit_flag", False))
    if not overfit:
        return False
    # Optional extra safeguard: if even best 4h val is still weak
    return True


def description_from_metrics(metrics: Optional[Dict[str, Any]]) -> str:
    if not metrics:
        return "missing metrics"
    return (
        f"model={metrics.get('model_type')} "
        f"data={metrics.get('dataset_suffix', '') or '4h'} "
        f"val_char={metrics.get('best_val_acc_char')} "
        f"gap_char={metrics.get('gap_char')} "
        f"overfit={metrics.get('overfit_flag')}"
    )


# =========================
# Main loop
# =========================

def main() -> None:
    ensure_results_tsv()
    ensure_logs_dir()

    if not file_exists(PROGRAM_FILE):
        raise FileNotFoundError(f"Missing {PROGRAM_FILE}")
    if not file_exists(TRAIN_FILE):
        raise FileNotFoundError(f"Missing {TRAIN_FILE}")

    log_loop(f"[loop] branch={get_head_branch()} commit={get_head_commit()}")

    best_metrics_global: Optional[Dict[str, Any]] = None
    best_metrics_4h: Optional[Dict[str, Any]] = None

    no_improvement_count = 0
    total_trials = 0
    four_h_trials = 0

    current_dataset_suffix = ""

    while total_trials < MAX_TOTAL_EXPERIMENTS:
        if file_exists(STOP_FILE):
            log_loop("[loop] stop file detected, exiting gracefully.")
            break

        # Escalate to 40h if needed
        if current_dataset_suffix == "" and should_switch_to_40h(four_h_trials, best_metrics_4h, best_metrics_4h):
            log_loop("[loop] switching from 4h to 40h")
            changed = set_dataset_suffix_in_train_file("40h")
            if changed:
                git_commit_all("chore: switch dataset from 4h to 40h")
            current_dataset_suffix = "40h"

        # Build Aider prompt
        if best_metrics_global is None:
            prompt = (
                "Read program.md and train_gawf.py.\n"
                "Do not restate the files.\n"
                "Prepare the first experiment according to program.md.\n"
                "Only modify train_gawf.py.\n"
                "Make one small change or keep the current baseline if already appropriate."
            )
        else:
            prompt = summarize_metrics(best_metrics_global)

        trial_index = total_trials + 1
        log_loop(f"[loop] trial={trial_index}, dataset={current_dataset_suffix or '4h'}")
        log_loop("[loop] calling aider...")
        call_aider(prompt, trial_index)
        log_loop("[loop] aider finished.")

        # Local commit after aider edit
        git_commit_all(f"exp: trial {total_trials + 1} propose next experiment")

        # Run training
        log_loop("[loop] starting training...")
        status, returncode = run_training(current_dataset_suffix)
        log_loop(f"[loop] training finished with status={status}")
        metrics = load_metrics_or_none()
        commit = get_head_commit()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        if status != "ok" or metrics is None:
            append_tsv_row(
                RESULTS_TSV,
                [
                    timestamp,
                    commit,
                    current_dataset_suffix,
                    "",
                    "0.0",
                    "0.0",
                    "0.0",
                    "0.0",
                    "999.0",
                    "true",
                    "crash" if status != "timeout" else "timeout",
                    f"training failed: status={status}, returncode={returncode}",
                ],
            )
            no_improvement_count += 1
            total_trials += 1
            if current_dataset_suffix == "":
                four_h_trials += 1
            log_loop(f"[loop] training failed: status={status}, returncode={returncode}")
            if no_improvement_count >= MAX_NO_IMPROVEMENT:
                log_loop("[loop] too many consecutive failures/no improvement, stopping.")
                break
            continue

        # Track bests
        improved = is_improved(metrics, best_metrics_global)
        if improved:
            best_metrics_global = metrics
            no_improvement_count = 0
            status_label = "keep"
        else:
            no_improvement_count += 1
            status_label = "discard"

        if current_dataset_suffix == "":
            four_h_trials += 1
            if best_metrics_4h is None or is_improved(metrics, best_metrics_4h):
                best_metrics_4h = metrics

        # Save results row
        append_tsv_row(
            RESULTS_TSV,
            [
                timestamp,
                commit,
                str(metrics.get("dataset_suffix", "")),
                str(metrics.get("model_type", "")),
                str(metrics.get("best_val_acc_char", "")),
                str(metrics.get("best_val_acc_pos", "")),
                str(metrics.get("final_train_acc_char", "")),
                str(metrics.get("final_val_acc_char", "")),
                str(metrics.get("gap_char", "")),
                str(metrics.get("overfit_flag", "")),
                status_label,
                description_from_metrics(metrics),
            ],
        )

        maybe_commit_results_snapshot()
        total_trials += 1

        log_loop(
            "[loop] done "
            f"trial={total_trials} "
            f"status={status_label} "
            f"val_char={metrics.get('best_val_acc_char')} "
            f"gap_char={metrics.get('gap_char')} "
            f"overfit={metrics.get('overfit_flag')}"
        )

        # Stop conditions
        if no_improvement_count >= MAX_NO_IMPROVEMENT:
            log_loop("[loop] no improvement limit reached, stopping.")
            break

    log_loop("[loop] finished.")


if __name__ == "__main__":
    main()