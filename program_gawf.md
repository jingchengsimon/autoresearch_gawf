# GAWF autoresearch program

Editable file: train_gawf.py only.

Prefer small, controlled changes.
Avoid large simultaneous modifications.
You must change at least one hyperparameter in train_gawf.py. If no change is made, the experiment is invalid.

Do not change from sector mode to coord mode unless explicitly instructed.
Current autoresearch must stay in sector mode.

Do NOT rename, delete, or restyle existing variables, function arguments, config keys, helper names, metric field names, or logging field names unless absolutely necessary.
Prefer changing only values, not names.

Dataset:
Use only the 40h dataset.
Set dataset_suffix = "40h".

Current objective:
Do NOT focus on overfitting reduction for now.
The goal is to minimize validation loss, with the following priority:

1. lowest val_char_loss
2. lowest val_pos_loss

Primary ranking metric:
- lower val_char_loss is always better

Secondary ranking metric:
- if val_char_loss is similar, prefer lower val_pos_loss

If both losses are similar, prefer the simpler and smaller change.

Allowed modifications:
- model_type (rnn / gru / lstm / gawf / ffn / dann)
- hidden_size
- learning_rate
- optimizer
- weight_decay
- dropout_rate
- num_epochs
- feedback parameters for gawf only

Restrictions:
1. num_epochs must not exceed 30
2. if model_type != "gawf", do not touch FB_START_EPOCH
3. FB_START_EPOCH may be changed only during GAWF optimization

Research phases:
Phase 1: RNN optimization
Phase 2: GAWF optimization
Phase 3: FFN / DANN comparison

Success criteria:
A modification is considered successful if it reduces val_char_loss.
If val_char_loss is effectively unchanged, then a lower val_pos_loss is preferred.

Avoid changes that do not improve either val_char_loss or val_pos_loss.

When proposing the next experiment:
- modify only train_gawf.py
- make exactly one small experimental change
- prioritize val_char_loss
- use val_pos_loss as the second criterion
- do not restate files
- do not explain the whole program
- directly apply the code change