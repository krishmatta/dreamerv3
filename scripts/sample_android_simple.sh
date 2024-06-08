#! /bin/bash

task=$1
seed=$2

shift
shift

python dreamerv3/train.py \
       --run.script sample \
       --logdir ~/logdir/android_simple/$name \
       --task $task \
       --seed $seed \
       --jax.platform cpu \
       --batch_size 8 \
       --batch_length 12 \
       --run.steps 100 \
       --run.train_ratio 32 \
       "$@"
