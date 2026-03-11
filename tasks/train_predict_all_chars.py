"""
Encapsulates training and evaluation logic for predict_all_chars mode.
Avoids scattered if-else in the main training script.
"""
from collections import Counter
import torch
import torch.nn.functional as F


def loss_char_all_chars(out_char, labels, criterion_char, max_chars, device):
    """
    Character loss for all-chars mode: greedy matching + CE per matched pair.
    out_char: (B, T, max_chars, num_classes), labels: (B, T, max_chars).
    Returns: loss_char (scalar tensor), loss_pos is always 0 in this mode.
    """
    batch_size, frame_num, max_chars_pred, num_classes = out_char.shape
    pred_probs = F.softmax(out_char, dim=-1)
    total_loss = 0.0
    total_valid_chars = 0

    for b in range(batch_size):
        for t in range(frame_num):
            true_chars = labels[b, t]
            valid_mask = true_chars >= 0
            valid_true_chars = true_chars[valid_mask]
            if len(valid_true_chars) == 0:
                continue

            frame_probs = pred_probs[b, t]
            frame_logits = out_char[b, t]
            used_mask = torch.zeros(max_chars, dtype=torch.bool, device=device)
            matched_pred_indices = []
            matched_true_chars = []

            for true_char_id in valid_true_chars:
                char_probs = frame_probs[:, true_char_id].clone()
                char_probs = char_probs.masked_fill(used_mask, -1.0)
                best_pred_idx = torch.argmax(char_probs).item()
                matched_pred_indices.append(best_pred_idx)
                matched_true_chars.append(true_char_id)
                used_mask[best_pred_idx] = True

            if len(matched_pred_indices) > 0:
                matched_pred_indices_tensor = torch.tensor(matched_pred_indices, device=device)
                matched_true_chars_tensor = torch.tensor(matched_true_chars, dtype=torch.long, device=device)
                matched_logits = frame_logits[matched_pred_indices_tensor]
                batch_loss = criterion_char(matched_logits, matched_true_chars_tensor)
                total_loss += batch_loss
                total_valid_chars += len(matched_pred_indices)

    if total_valid_chars == 0:
        loss_char = torch.tensor(0.0, device=device)
    else:
        loss_char = total_loss / total_valid_chars
    loss_pos = torch.tensor(0.0, device=device)
    return loss_char, loss_pos


def batch_metrics_all_chars(out_char, labels, max_chars, device):
    """
    Training batch metrics: exact frame match (scheme B). Call only when batch_idx % 50 == 0 or last batch.
    Returns: (batch_exact, batch_frames_eval) to add to epoch_train_acc_char and epoch_train_frames_eval.
    """
    batch_size, frame_num = labels.shape[:2]
    batch_exact = 0
    batch_frames_eval = 0
    pred_probs = F.softmax(out_char, dim=-1)

    for b in range(batch_size):
        for t in range(frame_num):
            true_chars = labels[b, t]
            valid_mask = true_chars >= 0
            valid_true_chars = true_chars[valid_mask]
            if len(valid_true_chars) == 0:
                continue
            batch_frames_eval += 1

            frame_probs = pred_probs[b, t]
            frame_logits = out_char[b, t]
            used_mask = torch.zeros(max_chars, dtype=torch.bool, device=device)
            matched_pred_chars = []

            for true_char_id in valid_true_chars:
                char_probs = frame_probs[:, int(true_char_id)].clone()
                char_probs = char_probs.masked_fill(used_mask, -1.0)
                best_pred_idx = torch.argmax(char_probs).item()
                pred_char = torch.argmax(frame_logits[best_pred_idx]).item()
                matched_pred_chars.append(pred_char)
                used_mask[best_pred_idx] = True

            gt_chars = [int(c.item()) if isinstance(c, torch.Tensor) else int(c) for c in valid_true_chars]
            if Counter(matched_pred_chars) == Counter(gt_chars):
                batch_exact += 1

    return batch_exact, batch_frames_eval


def eval_accumulate_batch_all_chars(out_char, labels, device):
    """
    Eval loop: accumulate (total_frames_exact, total_frames_eval) for one batch.
    Returns: (batch_exact, batch_frames_eval).
    """
    batch_size, frame_num = labels.shape[:2]
    pred_probs = F.softmax(out_char, dim=-1)
    batch_exact = 0
    batch_frames_eval = 0

    for b in range(batch_size):
        for t in range(frame_num):
            true_chars = labels[b, t]
            valid_mask = true_chars >= 0
            valid_true_chars = true_chars[valid_mask]
            if len(valid_true_chars) == 0:
                continue
            batch_frames_eval += 1

            frame_probs = pred_probs[b, t]
            frame_logits = out_char[b, t]
            used_mask = torch.zeros(frame_logits.shape[0], dtype=torch.bool, device=device)
            matched_pred_chars = []

            for true_char_id in valid_true_chars:
                char_probs = frame_probs[:, int(true_char_id)].clone()
                char_probs = char_probs.masked_fill(used_mask, -1.0)
                best_pred_idx = torch.argmax(char_probs).item()
                pred_char = torch.argmax(frame_logits[best_pred_idx]).item()
                matched_pred_chars.append(pred_char)
                used_mask[best_pred_idx] = True

            gt_chars = [int(c.item()) for c in valid_true_chars]
            if Counter(matched_pred_chars) == Counter(gt_chars):
                batch_exact += 1

    return batch_exact, batch_frames_eval


def finalize_acc_all_chars(total_exact, total_eval):
    """Convert accumulated (exact, eval) to accuracy percentage. metric_pos is 0."""
    acc_char = (total_exact / total_eval) * 100 if total_eval > 0 else 0.0
    metric_pos = 0.0
    return acc_char, metric_pos


def loss_weights_all_chars():
    return [1, 0]


def build_loss_fn_all_chars(mdl, criterion_char, max_chars, device, loss_weights, rnn_diag_lambda):
    """Build a single loss_fn(out_char, out_pos, labels) for all-chars mode (includes RNN diag and weights)."""
    def loss_fn(out_char, out_pos, labels):
        loss_char, loss_pos = loss_char_all_chars(out_char, labels, criterion_char, max_chars, device)
        if hasattr(mdl, 'rnn') and mdl.rnn is not None:
            rnn_hh_diag = mdl.rnn.weight_hh_l0.diagonal().abs().mean()
        else:
            rnn_hh_diag = torch.tensor(0.0, device=device)
        return (loss_weights[0] * loss_char + loss_weights[1] * loss_pos +
                rnn_diag_lambda * rnn_hh_diag)
    return loss_fn


class AllCharsMetricsMode:
    """
    Encapsulates all predict_all_chars metrics state and logic.
    Single place for total_frames_exact / total_frames_eval and train/eval accumulation.
    """

    def __init__(self, max_chars, device):
        self.max_chars = max_chars
        self.device = device

    def init_epoch_train(self):
        return {"exact": 0, "eval": 0}

    def update_train_batch(self, acc, out_char, labels, batch_idx, len_train_dl, out_pos=None):
        if batch_idx % 50 != 0 and batch_idx != len_train_dl - 1:
            return acc
        batch_exact, batch_frames_eval = batch_metrics_all_chars(
            out_char, labels, self.max_chars, self.device
        )
        if batch_frames_eval > 0:
            acc["exact"] += batch_exact
            acc["eval"] += batch_frames_eval
        return acc

    def finalize_train_epoch(self, acc, num_batches=None):
        train_acc_char = (acc["exact"] / acc["eval"]) * 100 if acc["eval"] > 0 else 0.0
        train_metric_pos = 0.0
        return train_acc_char, train_metric_pos

    def init_eval(self):
        return {"exact": 0, "eval": 0}

    def update_eval_batch(self, acc, out_char, labels):
        be, bf = eval_accumulate_batch_all_chars(out_char, labels, self.device)
        acc["exact"] += be
        acc["eval"] += bf
        return acc

    def finalize_eval(self, acc, num_batches):
        return finalize_acc_all_chars(acc["exact"], acc["eval"])

    def format_train_str(self, epoch, num_epochs, acc_char, metric_pos, gpu_info=""):
        return f"Epoch {epoch + 1}/{num_epochs} - Train (all chars acc): {acc_char:.2f}%{gpu_info}"

    def format_val_str(self, acc_char, metric_pos):
        return f" Validation (all chars acc): {acc_char:.2f}%"

    def postfix_for_pbar(self, current_loss, out_char, out_pos, labels):
        return {"loss": f"{current_loss:.4f}"}

    def add_pos_to_result_dict(self, base, train_metric_pos, val_metric_pos, actual_epochs,
                                train_loss_pos=None, val_loss_pos=None,
                                train_loss_char=None, val_loss_char=None):
        """All-chars has no pos/char-loss keys; return base unchanged."""
        return base
