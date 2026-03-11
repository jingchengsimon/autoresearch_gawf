# GAWF autoresearch protocol

Goal: maximize validation accuracy while minimizing overfitting.

Editable file:
- train_gawf.py

Read-only files:
- prepare_gawf.py
- engine/*
- models/*
- tasks/*
- utils/*

Only modify train_gawf.py.

Dataset policy:
- default dataset: 4h (no suffix)
- training schedule: 50 epochs
- if repeated RNN experiments show persistent overfitting, switch to 40h float32 dataset
- 40h runs are slower and should be used sparingly

Research phase order (do not skip):
1. RNN optimization
2. GAWF optimization
3. FFN / DANN optimization

Phase 1: RNN
Models: rnn, gru, lstm  
Goal: establish strong baseline and reduce overfitting

Search space:
- hidden_size
- learning_rate
- dropout
- weight_decay
- optimizer

Phase 2: GAWF
Focus on feedback mechanisms.

Search space:
- hidden_size
- learning_rate
- dropout
- weight_decay
- nofb
- fb_start_epoch
- optimizer

Phase 3: FFN / DANN
Focus on model capacity and regularization.

Search space:
- hidden_size
- dropout
- learning_rate
- optimizer

Overfitting signals:
- training accuracy increases while validation stagnates or drops
- validation loss increases
- large train–validation gap

Experiment rules:
- make only small changes each experiment
- prefer simpler solutions
- discard changes that add complexity with little gain

Training command:
python train_gawf.py

Each run outputs metrics (e.g. val_acc).

Record experiments in results.tsv:

commit	val_acc	memory_gb	status	description