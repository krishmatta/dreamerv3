#! /bin/bash

task=$1
seed=$2

shift
shift

python dreamerv3/main.py \
       --run.script sample \
       --logdir ~/logdir/android/$name \
       --task $task \
       --seed $seed \
       --batch_size 16 \
       --batch_length 256 \
       --run.train_ratio 32 \
       --run.steps 100 \
       --run.driver_parallel False \
       --run.num_envs 1 \
       "$@"
