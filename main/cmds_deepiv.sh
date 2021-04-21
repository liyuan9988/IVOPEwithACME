#!/bin/bash

DATASET_PATH="SET_PATH_TO_THE_DATASET"

## Near-policy datasets.

# Valid values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
NOISE_LEVEL=0.0

# bsuite_cartpole
python run_deepiv.py \
  --density_layer_sizes=32,32 \
  --density_learning_rate=0.0001 \
  --num_cat=10 \
  --problem_config.near_policy_dataset=True \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_cartpole \
  --value_layer_sizes=50,50 \
  --value_learning_rate=0.003 \
  --dataset_path="$DATASET_PATH"

# bsuite_catch
python run_deepiv.py \
  --density_layer_sizes=128,128 \
  --density_learning_rate=1e-05 \
  --num_cat=1 \
  --problem_config.near_policy_dataset=True \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_catch \
  --value_layer_sizes=50,50 \
  --value_learning_rate=0.003 \
  --dataset_path="$DATASET_PATH"

# bsuite_mountain_car
python run_deepiv.py \
  --density_layer_sizes=64,64 \
  --density_learning_rate=1e-05 \
  --num_cat=3 \
  --problem_config.near_policy_dataset=True \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_mountain_car \
  --value_layer_sizes=50,50 \
  --value_learning_rate=0.001 \
  --dataset_path="$DATASET_PATH"


## Pure offline datasets. BSuite tasks.

# Valid values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
NOISE_LEVEL=0.0

# bsuite_cartpole
python run_deepiv.py \
  --density_layer_sizes=128,128 \
  --density_learning_rate=3e-05 \
  --num_cat=10 \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_cartpole \
  --value_layer_sizes=50,50 \
  --value_learning_rate=0.001 \
  --dataset_path="$DATASET_PATH"

# bsuite_catch
python run_deepiv.py \
  --density_layer_sizes=128,128 \
  --density_learning_rate=1e-05 \
  --num_cat=3 \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_catch \
  --value_layer_sizes=50,50 \
  --value_learning_rate=0.003 \
  --dataset_path="$DATASET_PATH"

# bsuite_mountain_car
python run_deepiv.py \
  --density_layer_sizes=128,128 \
  --density_learning_rate=1e-05 \
  --num_cat=3 \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=bsuite_mountain_car \
  --value_layer_sizes=50,50 \
  --value_learning_rate=0.003 \
  --dataset_path="$DATASET_PATH"


## Pure offline datasets. DM Control Suite tasks.

# Valid values: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
NOISE_LEVEL=0.0

# dm_control_cartpole_swingup
python run_deepiv.py \
  --density_layer_sizes=768,768,384 \
  --density_learning_rate=0.0003 \
  --num_cat=3 \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=dm_control_cartpole_swingup \
  --value_layer_sizes=512,512,256 \
  --value_learning_rate=0.0003 \
  --dataset_path="$DATASET_PATH"

# dm_control_cheetah_run
python run_deepiv.py \
  --density_layer_sizes=768,768,384 \
  --density_learning_rate=0.0003 \
  --num_cat=10 \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=dm_control_cheetah_run \
  --value_layer_sizes=512,512,256 \
  --value_learning_rate=0.001 \
  --dataset_path="$DATASET_PATH"

# dm_control_walker_walk
python run_deepiv.py \
  --density_layer_sizes=768,768,384 \
  --density_learning_rate=0.0003 \
  --num_cat=10 \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=dm_control_walker_walk \
  --value_layer_sizes=512,512,256 \
  --value_learning_rate=0.001 \
  --dataset_path="$DATASET_PATH"

# dm_control_humanoid_run
python run_deepiv.py \
  --density_layer_sizes=768,768,384 \
  --density_learning_rate=0.001 \
  --num_cat=10 \
  --problem_config.noise_level=$NOISE_LEVEL \
  --problem_config.task_name=dm_control_humanoid_run \
  --value_layer_sizes=512,512,256 \
  --value_learning_rate=0.001 \
  --dataset_path="$DATASET_PATH"
