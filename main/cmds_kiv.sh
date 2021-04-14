#!/bin/bash

DATASET_PATH="SET_PATH_TO_THE_DATASET"

## Pure offline datasets. BSuite tasks.

# Valid values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
NOISE_LEVEL=0.0

# The hyper-parameters if KIV is grid-searched for every environment and every
# random noise level.
#
# Range of stage1_reg: 1e-8, 1e-6, 1e-4, 1e-2
STAGE1_REG=1e-8
# Range of stage2_reg: 1e-8, 1e-6, 1e-4, 1e-2
STAGE2_REG=1e-8
# Range of n_component: 128, 256, 512, 1024
N_COMPONENT=1024

# bsuite_cartpole
python run_kiv.py \
  --n_component=$N_COMPONENT \
  --stage1_reg=$STAGE1_REG \
  --stage2_reg=$STAGE2_REG \
  --problem_config.prob_param.noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.env_noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.policy_noise_level=0.1 \
  --problem_config.task_name=bsuite_cartpole \
  --dataset_path="$DATASET_PATH"

# bsuite_catch
python run_kiv.py \
  --n_component=$N_COMPONENT \
  --stage1_reg=$STAGE1_REG \
  --stage2_reg=$STAGE2_REG \
  --problem_config.prob_param.noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.env_noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.policy_noise_level=0.1 \
  --problem_config.task_name=bsuite_catch \
  --dataset_path="$DATASET_PATH"

# bsuite_mountain_car
python run_kiv.py \
  --n_component=$N_COMPONENT \
  --stage1_reg=$STAGE1_REG \
  --stage2_reg=$STAGE2_REG \
  --problem_config.prob_param.noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.env_noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.policy_noise_level=0.1 \
  --problem_config.task_name=bsuite_mountain_car \
  --dataset_path="$DATASET_PATH"




## Pure offline datasets. DM Control Suite tasks.

# Valid values: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
NOISE_LEVEL=0.0

# The hyper-parameters if KIV is grid-searched for every environment and every
# random noise level.
#
# Range of stage1_reg: 1e-8, 1e-6, 1e-4, 1e-2
STAGE1_REG=1e-8
# Range of stage2_reg: 1e-8, 1e-6, 1e-4, 1e-2
STAGE2_REG=1e-8
# Range of n_component: 128, 256, 512, 1024
N_COMPONENT=1024

# dm_control_cartpole_swingup
python run_kiv.py \
  --n_component=$N_COMPONENT \
  --stage1_reg=$STAGE1_REG \
  --stage2_reg=$STAGE2_REG \
  --problem_config.prob_param.noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.env_noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.policy_noise_level=0.2 \
  --problem_config.task_name=dm_control_cartpole_swingup \
  --dataset_path="$DATASET_PATH"

# dm_control_cheetah_run
python run_kiv.py \
  --n_component=$N_COMPONENT \
  --stage1_reg=$STAGE1_REG \
  --stage2_reg=$STAGE2_REG \
  --problem_config.prob_param.noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.env_noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.policy_noise_level=0.2 \
  --problem_config.task_name=dm_control_cheetah_run \
  --dataset_path="$DATASET_PATH"

# dm_control_walker_walk
python run_kiv.py \
  --n_component=$N_COMPONENT \
  --stage1_reg=$STAGE1_REG \
  --stage2_reg=$STAGE2_REG \
  --problem_config.prob_param.noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.env_noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.policy_noise_level=0.2 \
  --problem_config.task_name=dm_control_walker_walk \
  --dataset_path="$DATASET_PATH"

# dm_control_humanoid_run
python run_kiv.py \
  --n_component=$N_COMPONENT \
  --stage1_reg=$STAGE1_REG \
  --stage2_reg=$STAGE2_REG \
  --problem_config.prob_param.noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.env_noise_level=$NOISE_LEVEL \
  --problem_config.target_policy_param.policy_noise_level=0.2 \
  --problem_config.task_name=dm_control_humanoid_run \
  --dataset_path="$DATASET_PATH"
