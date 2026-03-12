# GAWF autoresearch program

Editable file: train_gawf.py only. Prefer small, controlled changes. Avoid large simultaneous modifications.

Do not change from sector mode to coord mode unless explicitly instructed.
Current autoresearch should stay in sector mode.

Do NOT rename or delete existing variables, function arguments, config keys, helper names, or logging field names unless absolutely necessary.

Prefer changing only values, not names. 

You must change at least one hyperparameter in train_gawf.py. If no change is made the experiment is invalid.

Datasets:
Stage 1: 4h dataset (default, no suffix)
Stage 2: 40h dataset (suffix="40h")

Primary objective:
Improve validation character accuracy (val_acc_char) while keeping training accuracy strong and reducing the train–validation generalization gap.

Metric definitions:

gap_char = train_acc_char - val_acc_char
gap_pos  = train_acc_pos  - val_acc_pos

Success criteria:

A modification is considered successful only if:

1. val_acc_char increases
   OR stays within ±0.5% of the previous best

2. train_acc_char does NOT decrease by more than 1.0%

3. preference is given to configurations that reduce gap_char

Experiment ranking priority:

1. higher val_acc_char
2. smaller gap_char
3. stable train_acc_char
4. secondary metric: val_acc_pos

Avoid solutions that improve validation only by collapsing training accuracy.

Allowed modifications:

model_type (rnn / gru / lstm / gawf)
hidden_size
learning_rate
optimizer
weight_decay
dropout_rate
num_epochs
feedback parameters (for gawf)

NOTICE:
1. num_epochs should not exceed 30.
2. If model_type != "gawf", do not touch FB_START_EPOCH.

Overfitting detection:

overfit_flag = true if gap_char > 10 OR gap_pos > 10

Dataset escalation rule:

If 10 distinct experiments on the 4h dataset still show large overfitting
and val_acc_char does not significantly improve,
stop searching on 4h and switch to the 40h dataset.

Research phases:

Phase 1: RNN optimization
Phase 2: GAWF optimization
Phase 3: FFN / DANN comparison