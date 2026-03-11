# GAWF autoresearch – core rules

- only modify `train_gawf.py`
- 4h data first, 50 epochs
- phase order: RNN → GAWF → FFN
- if persistent overfit on RNN, switch to 40h (float32) data
# Legacy detailed spec (can be ignored by aider)

This repository runs an autoresearch loop where an LLM autonomously proposes and tests model improvements.

The goal is to improve generalization performance on the AIM3 sequence task while controlling overfitting.

This program follows the general autoresearch protocol by Karpathy, but includes task-specific constraints and research phases.

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the codebase**：Read the following files for full context:
prepare_gawf.py
train_gawf.py
engine/*
models/*
tasks/*
utils/*
Important rule: prepare_gawf.py and files in these 4 folders are read-only, train_gawf.py is editable
4. **Verify dataset availability**: Check that the dataset exists and is accessible. The system supports two datasets: 4h dataset (default), 40h dataset.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. Training command: `python train_gawf.py`. Each run trains the model and produces a metric summary.

## Allowed modifications

The agent may modify ONLY: train_gawf.py. Inside this file it may change:
model type  
hidden size  
optimizer  
learning rate  
weight decay  
dropout  
feedback configuration  
training schedule  
hyperparameters  
experiment logic

## Forbidden modifications

The agent must NOT modify:
prepare_gawf.py  
engine/*  
models/*  
tasks/*  
utils/*  

dataset definitions  
evaluation metrics  

The evaluation pipeline is fixed.

## Research Goal

The primary goal is: maximize validation accuracy while minimizing overfitting.

Key signals:
validation accuracy  
train vs validation gap  
validation loss trends  
stability across runs

## Dataset Policy

The system uses two dataset sizes.

### Stage 1 dataset

Default dataset: 4h dataset(No data suffix).
Training schedule: 50 epochs. 
Each experiment runs the full training.

### Stage 2 dataset escalation

If multiple RNN experiments fail to reduce overfitting, switch to: 40h dataset float32

Reasons: determine whether overfitting is caused by insufficient data   validate promising architecture ideas, continue research once 4h dataset becomes limiting

40h runs are slower (~4 hours). Use them sparingly.

## Overfitting Decision Rule

A model is considered overfitting when: train accuracy continues increasing validation accuracy stagnates or decreases validation loss worsens train-validation gap remains large

If these symptoms persist across several RNN experiments: stop spending search budget on the 4h dataset and switch to 40h dataset

## Research Phases

Experiments must follow the model research order below. Do NOT jump ahead.

## Phase 1 – RNN Optimization

Start with baseline recurrent models. Model types: rnn, gru, lstm  
Focus on:
reducing overfitting  
improving validation accuracy  
stabilizing learning dynamics  

Search space includes:
hidden_size  
learning_rate  
dropout  
weight_decay  
optimizer  
training schedule  

Goal: establish a strong RNN baseline.

## Phase 2 – GAWF Optimization

Once a stable RNN baseline exists, move to: gawf

Focus on:
feedback schedule  
feedback gating behavior  
regularization  
hyperparameters  

Search space includes:
hidden_size  
lr  
dropout  
weight_decay  
nofb  
fb_start_epoch  
optimizer  

Comparisons should remain fair relative to the best RNN baseline.

## Phase 3 – ANN / DANN / FFN

After RNN and GAWF experiments: do optimization on ffn and dann  

Focus on:
model capacity  
hidden_size  
dropout  
optimizer  
learning rate  

Goal: determine whether feedforward models can compete with recurrent architectures.

## Simplicity Criterion

When evaluating changes: simpler solutions are preferred.

Examples:

GOOD:
removing unnecessary complexity  
achieving equal performance with fewer parameters  

BAD:
large code complexity for tiny improvements  

Tradeoff rule:
small improvement + large complexity = discard  
small improvement + simpler code = keep  

## Output Format

After each run the script produces a metric summary.

Example:
val_acc: 0.842  
training_seconds: 1400  
peak_vram_mb: 19000  
num_params_M: 12.3  

## Logging Results

Each experiment must be recorded in results.tsv

Format:

commit	val_acc	memory_gb	status	description

Example:
```
commit	val_acc	memory_gb	status	description  
a1b2c3d	0.842	18.2	keep	baseline rnn  
b2c3d4e	0.851	18.4	keep	lr=0.0005  
c3d4e5f	0.835	18.2	discard	increase dropout  
d4e5f6g	0.000	0.0	crash	hidden size 2048 OOM  
```

## Experiment Loop

LOOP FOREVER:

1. Inspect current git state
2. Modify train_gawf.py
3. Commit the change
4. Run experiment
5. Extract metrics
6. Log results
7. Keep commit if metric improved
8. Otherwise revert

## Timeout

If a run exceeds reasonable runtime, treat it as failure.

Recommended target runtime:
4h dataset runs: 30–60 minutes  
40h dataset runs: 5 hours

## Crashes

If run crashes:
inspect stack trace  
attempt fix  
if idea is fundamentally broken, log crash and move on  

## Autonomy Rule

Once the experiment loop starts: DO NOT stop to ask the human. Continue experiments indefinitely until manually interrupted. The system is designed to run autonomously for long periods. Example scenario: If each experiment takes ~30 minutes, the system can run 16–24 experiments overnight. The human may review results later.
