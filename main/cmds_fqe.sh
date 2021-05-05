#!/bin/bash

DATASET_PATH="SET_PATH_TO_THE_DATASET"

## Near-policy datasets.

# Valid values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
NOISE_LEVEL=0.0
# Distributional Q network if True.
DISTRIBUTIONAL=True

# bsuite_cartpole
python run_fqe.py \
  --problem_config.near_policy_dataset=True \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_cartpole \
  --learning_rate=0.0003 \
  --target_update_period=100 \
  --distributional=$DISTRIBUTIONAL \
  --layer_sizes="50,50" \
  --dataset_path="$DATASET_PATH"

# bsuite_catch
python run_fqe.py \
  --problem_config.near_policy_dataset=True \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_catch \
  --learning_rate=1e-05 \
  --target_update_period=100 \
  --distributional=$DISTRIBUTIONAL \
  --layer_sizes="50,50" \
  --dataset_path="$DATASET_PATH"

# bsuite_mountain_car
python run_fqe.py \
  --problem_config.near_policy_dataset=True \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_mountain_car \
  --learning_rate=3e-05 \
  --target_update_period=100 \
  --distributional=$DISTRIBUTIONAL \
  --layer_sizes="50,50" \
  --dataset_path="$DATASET_PATH"


## Pure offline datasets. BSuite tasks.

# Valid values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
NOISE_LEVEL=0.0
# Distributional Q network if True.
DISTRIBUTIONAL=True

# bsuite_cartpole
python run_fqe.py \
  --learning_rate=0.0001 \
  --target_update_period=100 \
  --distributional=$DISTRIBUTIONAL \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_cartpole \
  --layer_sizes="50,50" \
  --dataset_path="$DATASET_PATH"

# bsuite_catch
python run_fqe.py \
  --learning_rate=1e-05 \
  --target_update_period=100 \
  --distributional=$DISTRIBUTIONAL \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_catch \
  --layer_sizes="50,50" \
  --dataset_path="$DATASET_PATH"

# bsuite_mountain_car
python run_fqe.py \
  --learning_rate=0.0003 \
  --target_update_period=100 \
  --distributional=$DISTRIBUTIONAL \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_mountain_car \
  --layer_sizes="50,50" \
  --dataset_path="$DATASET_PATH"


## Pure offline datasets. DM Control Suite tasks.

# Valid values: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
NOISE_LEVEL=0.0
# Distributional Q network if True.
DISTRIBUTIONAL=True

# dm_control_cartpole_swingup
python run_fqe.py \
  --learning_rate=0.0003 \
  --target_update_period=100 \
  --distributional=$DISTRIBUTIONAL \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=dm_control_cartpole_swingup \
  --layer_sizes="512,512,256" \
  --dataset_path="$DATASET_PATH"

# dm_control_cheetah_run
python run_fqe.py \
  --learning_rate=3e-05 \
  --target_update_period=100 \
  --distributional=$DISTRIBUTIONAL \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=dm_control_cheetah_run \
  --layer_sizes="512,512,256" \
  --dataset_path="$DATASET_PATH"

# dm_control_humanoid_run
python run_fqe.py \
  --learning_rate=0.0003 \
  --target_update_period=100 \
  --distributional=$DISTRIBUTIONAL \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=dm_control_humanoid_run \
  --layer_sizes="512,512,256" \
  --dataset_path="$DATASET_PATH"

# dm_control_walker_walk
python run_fqe.py \
  --learning_rate=1e-05 \
  --target_update_period=100 \
  --distributional=$DISTRIBUTIONAL \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=dm_control_walker_walk \
  --layer_sizes="512,512,256" \
  --dataset_path="$DATASET_PATH"
