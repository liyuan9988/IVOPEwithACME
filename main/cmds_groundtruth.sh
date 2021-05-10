#!/bin/bash

DATASET_PATH="SET_PATH_TO_THE_DATASET"

## Near-policy datasets.

# Valid values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
NOISE_LEVEL=0.0

# bsuite_cartpole
python run_groundtruth.py \
  --problem_config.near_policy_dataset=True \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_cartpole \
  --dataset_path="$DATASET_PATH"

# bsuite_catch
python run_groundtruth.py \
  --problem_config.near_policy_dataset=True \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_catch \
  --dataset_path="$DATASET_PATH"

# bsuite_mountain_car
python run_groundtruth.py \
  --problem_config.near_policy_dataset=True \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_mountain_car \
  --dataset_path="$DATASET_PATH"


## Pure offline datasets. BSuite tasks.

# Valid values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
NOISE_LEVEL=0.0

# bsuite_cartpole
python run_groundtruth.py \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_cartpole \
  --dataset_path="$DATASET_PATH"

# bsuite_catch
python run_groundtruth.py \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_catch \
  --dataset_path="$DATASET_PATH"

# bsuite_mountain_car
python run_groundtruth.py \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_mountain_car \
  --dataset_path="$DATASET_PATH"


## Pure offline datasets. DM Control Suite tasks.

# Valid values: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
NOISE_LEVEL=0.0

# dm_control_cartpole_swingup
python run_groundtruth.py \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=dm_control_cartpole_swingup \
  --dataset_path="$DATASET_PATH"

# dm_control_cheetah_run
python run_groundtruth.py \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=dm_control_cheetah_run \
  --dataset_path="$DATASET_PATH"

# dm_control_humanoid_run
python run_groundtruth.py \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=dm_control_humanoid_run \
  --dataset_path="$DATASET_PATH"

# dm_control_walker_walk
python run_groundtruth.py \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=dm_control_walker_walk \
  --dataset_path="$DATASET_PATH"
