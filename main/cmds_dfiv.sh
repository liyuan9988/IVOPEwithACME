#!/bin/bash

DATASET_PATH="SET_PATH_TO_THE_DATASET"

## Pure offline datasets. BSuite tasks.

# Valid values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
NOISE_LEVEL=0.0

# bsuite_cartpole
python run_dfiv.py \
  --batch_size=2048 \
  --instrumental_layer_sizes=50,50 \
  --instrumental_learning_rate=0.003 \
  --instrumental_reg=0.0001 \
  --learner2=True \
  --max_steps=100000 \
  --problem_config.prob_param.noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.env_noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.policy_noise_level=0.1 \
  --problem_config.task_name=bsuite_cartpole \
  --stage1_reg=1e-06 \
  --stage2_reg=1e-08 \
  --value_layer_sizes=50,50 \
  --value_learning_rate=0.0003 \
  --value_reg=0.0001 \
  --value_layer_sizes=50,50 \
  --dataset_path="$DATASET_PATH"

# bsuite_catch
python run_dfiv.py \
  --batch_size=2048 \
  --instrumental_layer_sizes=150,150 \
  --instrumental_learning_rate=0.0003 \
  --instrumental_reg=1e-08 \
  --learner2=True \
  --max_steps=100000 \
  --problem_config.prob_param.noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.env_noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.policy_noise_level=0.1 \
  --problem_config.task_name=bsuite_catch \
  --stage1_reg=0.0001 \
  --stage2_reg=1e-08 \
  --value_layer_sizes=50,50 \
  --value_learning_rate=0.0003 \
  --value_reg=1e-06 \
  --value_layer_sizes=50,50 \
  --dataset_path="$DATASET_PATH"

# bsuite_mountain_car
python run_dfiv.py \
  --batch_size=2048 \
  --instrumental_layer_sizes=100,100 \
  --instrumental_learning_rate=0.0001 \
  --instrumental_reg=1e-06 \
  --learner2=True \
  --max_steps=100000 \
  --problem_config.prob_param.noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.env_noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.policy_noise_level=0.1 \
  --problem_config.task_name=bsuite_mountain_car \
  --stage1_reg=0.0001 \
  --stage2_reg=0.0001 \
  --value_layer_sizes=50,50 \
  --value_learning_rate=0.0003 \
  --value_reg=1e-08 \
  --value_layer_sizes=50,50 \
  --dataset_path="$DATASET_PATH"



## Pure offline datasets. DM Control Suite tasks.

# Valid values: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
NOISE_LEVEL=0.0

# dm_control_cartpole_swingup
python run_dfiv.py \
  --instrumental_layer_sizes=512,512,256 \
  --instrumental_learning_rate=1e-05 \
  --instrumental_reg=1e-08 \
  --learner2=True \
  --max_steps=100000 \
  --problem_config.prob_param.noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.env_noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.policy_noise_level=0.2 \
  --problem_config.task_name=dm_control_cartpole_swingup \
  --stage1_reg=1e-06 \
  --stage2_reg=0.01 \
  --value_layer_sizes=512,512,256 \
  --value_learning_rate=0.0003 \
  --value_reg=0.0001 \
  --dataset_path="$DATASET_PATH"

# dm_control_cheetah_run
python run_dfiv.py \
  --instrumental_layer_sizes=512,512,256 \
  --instrumental_learning_rate=1e-05 \
  --instrumental_reg=0.0001 \
  --learner2=True \
  --max_steps=100000 \
  --problem_config.prob_param.noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.env_noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.policy_noise_level=0.2 \
  --problem_config.task_name=dm_control_cheetah_run \
  --stage1_reg=1e-06 \
  --stage2_reg=0.01 \
  --value_layer_sizes=512,512,256 \
  --value_learning_rate=0.0003 \
  --value_reg=0.0001 \
  --dataset_path="$DATASET_PATH"

# dm_control_humanoid_run
python run_dfiv.py \
  --instrumental_layer_sizes=768,768,384 \
  --instrumental_learning_rate=0.0003 \
  --instrumental_reg=1e-06 \
  --learner2=True \
  --max_steps=100000 \
  --problem_config.prob_param.noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.env_noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.policy_noise_level=0.2 \
  --problem_config.task_name=dm_control_humanoid_run \
  --stage1_reg=0.01 \
  --stage2_reg=1e-06 \
  --value_layer_sizes=512,512,256 \
  --value_learning_rate=0.0003 \
  --value_reg=0.0001 \
  --dataset_path="$DATASET_PATH"

# dm_control_walker_walk
python run_dfiv.py \
  --instrumental_layer_sizes=768,768,384 \
  --instrumental_learning_rate=3e-05 \
  --instrumental_reg=1e-08 \
  --learner2=True \
  --max_steps=100000 \
  --problem_config.prob_param.noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.env_noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.policy_noise_level=0.2 \
  --problem_config.task_name=dm_control_walker_walk \
  --stage1_reg=0.01 \
  --stage2_reg=0.01 \
  --value_layer_sizes=512,512,256 \
  --value_learning_rate=0.0001 \
  --value_reg=1e-06 \
  --dataset_path="$DATASET_PATH"
