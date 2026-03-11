"""
Encapsulates training and evaluation logic for use_sector vs coordinate (single-char + position).
Avoids scattered if-else in the main training script.
"""
import torch
import torch.nn.functional as F


def loss_char_single(out_char, labels, criterion_char):
    """Character loss for single-char mode. labels[:,:,0] is char id."""
    labels_char = labels[:, :, 0].long().view(-1)
    outputs_char = out_char.reshape(-1, out_char.shape[-1])
    return criterion_char(outputs_char, labels_char)


def loss_pos_single(out_pos, labels, criterion_pos):
    """Position loss for sector classification."""
    labels_pos = labels[:, :, 1].long().view(-1)
    outputs_pos = out_pos.reshape(-1, out_pos.shape[-1])
    return criterion_pos(outputs_pos, labels_pos)


def batch_metric_char_single(out_char, labels):
    """Mean character correctness for one batch (single-char mode)."""
    return (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()


def batch_metric_pos_single(out_pos, labels):
    """Position metric for one batch: accuracy (sector)."""
    return (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()


def batch_loss_pos_sector(out_pos, labels):
    """Sector position cross-entropy loss for one batch (for logging/saving, like MSE in coord mode)."""
    labels_pos = labels[:, :, 1].long().view(-1)
    outputs_pos = out_pos.reshape(-1, out_pos.shape[-1])
    return F.cross_entropy(outputs_pos, labels_pos, reduction="mean").item()


def batch_loss_char_single(out_char, labels):
    """
    Character cross-entropy loss for one batch (single-char mode).
    Used only for logging/saving curves; always uses standard CE (same as criterion_char).
    """
    labels_char = labels[:, :, 0].long().view(-1)
    outputs_char = out_char.reshape(-1, out_char.shape[-1])
    return F.cross_entropy(outputs_char, labels_char, reduction="mean").item()


def eval_accumulate_batch_single(out_char, out_pos, labels):
    """
    Eval loop: accumulate total_acc_char and total_metric_pos for one batch (single-char).
    Returns: (acc_char_sum_delta, metric_pos_sum_delta).
    """
    acc_char = (torch.argmax(out_char, dim=2) == labels[:, :, 0].long()).float().mean().item()
    metric_pos = (torch.argmax(out_pos, dim=2) == labels[:, :, 1].long()).float().mean().item()
    return acc_char, metric_pos


def finalize_metrics_single(total_acc_char, total_metric_pos, num_batches):
    """Convert accumulated sums to (acc_char %, metric_pos %)."""
    acc_char = total_acc_char * 100 / num_batches
    metric_pos = total_metric_pos * 100 / num_batches
    return acc_char, metric_pos


def finalize_eval_metrics_single(total_acc_char, total_metric_pos, num_batches):
    """Eval: same as finalize_metrics_single."""
    return finalize_metrics_single(total_acc_char, total_metric_pos, num_batches)


def format_train_str_single(epoch, num_epochs, acc_char, metric_pos, gpu_info=""):
    return f"Epoch {epoch + 1}/{num_epochs} - Train (char, sector): ({acc_char:.2f}%, {metric_pos:.2f}%){gpu_info}"


def format_val_str_single(acc_char, metric_pos):
    return f" Validation (char, sector): ({acc_char:.2f}%, {metric_pos:.2f}%)"


def result_dict_keys_single():
    """Return the key names for the result dict (sector accuracy keys)."""
    return "train_acc_pos", "val_acc_pos"


def build_loss_fn_single(mdl, criterion_char, criterion_pos, loss_weights, rnn_diag_lambda, device):
    """Build a single loss_fn(out_char, out_pos, labels) for single-char + sector/coordinate (includes RNN diag)."""
    def loss_fn(out_char, out_pos, labels):
        loss_char = loss_char_single(out_char, labels, criterion_char)
        loss_pos = loss_pos_single(out_pos, labels, criterion_pos)
        if hasattr(mdl, 'rnn') and mdl.rnn is not None:
            rnn_hh_diag = mdl.rnn.weight_hh_l0.diagonal().abs().mean()
        else:
            rnn_hh_diag = torch.tensor(0.0, device=device)
        return (loss_weights[0] * loss_char + loss_weights[1] * loss_pos +
                rnn_diag_lambda * rnn_hh_diag)
    return loss_fn


class SingleCharMetricsMode:
    """
    Encapsulates single-char + sector/coordinate metrics state and logic.
    """

    def __init__(self, use_sector):
        self.use_sector = use_sector

    def init_epoch_train(self):
        d = {
            "acc_char_sum": 0.0,
            "metric_pos_sum": 0.0,
            "loss_char_sum": 0.0,
        }
        d["loss_pos_sum"] = 0.0
        return d

    def update_train_batch(self, acc, out_char, labels, batch_idx, len_train_dl, out_pos=None):
        acc["acc_char_sum"] += batch_metric_char_single(out_char, labels)
        acc["metric_pos_sum"] += batch_metric_pos_single(out_pos, labels)
        acc["loss_char_sum"] += batch_loss_char_single(out_char, labels)
        acc["loss_pos_sum"] += batch_loss_pos_sector(out_pos, labels)
        return acc

    def finalize_train_epoch(self, acc, num_batches):
        acc_char, metric_pos = finalize_metrics_single(
            acc["acc_char_sum"], acc["metric_pos_sum"], num_batches
        )
        loss_pos = (acc.get("loss_pos_sum", 0.0) / num_batches) if num_batches else None
        loss_char = (acc["loss_char_sum"] / num_batches) if num_batches else None
        return acc_char, metric_pos, loss_pos, loss_char

    def init_eval(self):
        d = {
            "acc_char_sum": 0.0,
            "metric_pos_sum": 0.0,
            "loss_char_sum": 0.0,
        }
        d["loss_pos_sum"] = 0.0
        return d

    def update_eval_batch(self, acc, out_char, labels, out_pos=None):
        ac, mp = eval_accumulate_batch_single(out_char, out_pos, labels)
        acc["acc_char_sum"] += ac
        acc["metric_pos_sum"] += mp
        acc["loss_pos_sum"] += batch_loss_pos_sector(out_pos, labels)
        acc["loss_char_sum"] += batch_loss_char_single(out_char, labels)
        return acc

    def finalize_eval(self, acc, num_batches):
        acc_char, metric_pos = finalize_eval_metrics_single(
            acc["acc_char_sum"], acc["metric_pos_sum"], num_batches
        )
        loss_pos = (acc.get("loss_pos_sum", 0.0) / num_batches) if num_batches else None
        loss_char = (acc["loss_char_sum"] / num_batches) if num_batches else None
        return acc_char, metric_pos, loss_pos, loss_char

    def format_train_str(self, epoch, num_epochs, acc_char, metric_pos, gpu_info=""):
        return format_train_str_single(epoch, num_epochs, acc_char, metric_pos, gpu_info)

    def format_val_str(self, acc_char, metric_pos):
        return format_val_str_single(acc_char, metric_pos)

    def postfix_for_pbar(self, current_loss, out_char, out_pos, labels):
        postfix = {"loss": f"{current_loss:.4f}"}
        pos_val = batch_metric_pos_single(out_pos, labels)
        postfix["pos_acc"] = f"{pos_val * 100:.2f}%"
        return postfix

    def add_pos_to_result_dict(self, base, train_metric_pos, val_metric_pos, actual_epochs,
                                train_loss_pos=None, val_loss_pos=None,
                                train_loss_char=None, val_loss_char=None):
        k1, k2 = result_dict_keys_single()
        base[k1] = train_metric_pos[:actual_epochs]
        base[k2] = val_metric_pos[:actual_epochs]
        if train_loss_pos is not None and val_loss_pos is not None:
            base["train_loss_pos"] = train_loss_pos[:actual_epochs]
            base["val_loss_pos"] = val_loss_pos[:actual_epochs]
        if train_loss_char is not None and val_loss_char is not None:
            base["train_loss_char"] = train_loss_char[:actual_epochs]
            base["val_loss_char"] = val_loss_char[:actual_epochs]
        return base
