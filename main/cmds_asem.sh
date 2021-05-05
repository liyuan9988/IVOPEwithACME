#!/bin/bash

DATASET_PATH="SET_PATH_TO_THE_DATASET"

## Near-policy datasets.

# Valid values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
NOISE_LEVEL=0.0

# bsuite_cartpole
python run_deep_gmm.py \
  --config.learner_class=AdversarialSEMLearner \
  --config.learner_config.AdversarialSEMLearner.clipping_action=False \
  --config.learner_config.AdversarialSEMLearner.critic_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.critic_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.critic_l2_regularizer=1e-10 \
  --config.learner_config.AdversarialSEMLearner.critic_lr=0.0003 \
  --config.learner_config.AdversarialSEMLearner.critic_regularizer=1e-08 \
  --config.learner_config.AdversarialSEMLearner.f_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.f_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.f_l2_regularizer=1e-08 \
  --config.learner_config.AdversarialSEMLearner.f_lr_multiplier=5.0 \
  --config.network_config.critic_layer_sizes=50,50 \
  --config.network_config.f_layer_sizes=150,150 \
  --config.problem_config.near_policy_dataset=True \
  --config.problem_config.noise_level=$NOISE_LEVEL \
  --config.problem_config.task_name=bsuite_cartpole \
  --dataset_path="$DATASET_PATH"

# bsuite_catch
python run_deep_gmm.py \
  --config.learner_class=AdversarialSEMLearner \
  --config.learner_config.AdversarialSEMLearner.clipping_action=False \
  --config.learner_config.AdversarialSEMLearner.critic_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.critic_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.critic_l2_regularizer=1e-08 \
  --config.learner_config.AdversarialSEMLearner.critic_lr=1e-05 \
  --config.learner_config.AdversarialSEMLearner.critic_regularizer=1e-10 \
  --config.learner_config.AdversarialSEMLearner.f_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.f_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.f_l2_regularizer=1e-06 \
  --config.learner_config.AdversarialSEMLearner.f_lr_multiplier=5.0 \
  --config.network_config.critic_layer_sizes=50,50 \
  --config.network_config.f_layer_sizes=100,100 \
  --config.problem_config.near_policy_dataset=True \
  --config.problem_config.noise_level=$NOISE_LEVEL \
  --config.problem_config.task_name=bsuite_catch \
  --dataset_path="$DATASET_PATH"

# bsuite_mountain_car
python run_deep_gmm.py \
  --config.learner_class=AdversarialSEMLearner \
  --config.learner_config.AdversarialSEMLearner.clipping_action=False \
  --config.learner_config.AdversarialSEMLearner.critic_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.critic_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.critic_l2_regularizer=1e-10 \
  --config.learner_config.AdversarialSEMLearner.critic_lr=0.001 \
  --config.learner_config.AdversarialSEMLearner.critic_regularizer=1e-08 \
  --config.learner_config.AdversarialSEMLearner.f_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.f_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.f_l2_regularizer=1e-10 \
  --config.learner_config.AdversarialSEMLearner.f_lr_multiplier=5.0 \
  --config.network_config.critic_layer_sizes=50,50 \
  --config.network_config.f_layer_sizes=100,100 \
  --config.problem_config.near_policy_dataset=True \
  --config.problem_config.noise_level=$NOISE_LEVEL \
  --config.problem_config.task_name=bsuite_mountain_car \
  --dataset_path="$DATASET_PATH"


## Pure offline datasets. BSuite tasks.

# Valid values: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
NOISE_LEVEL=0.0

# bsuite_cartpole
python run_deep_gmm.py \
  --config.learner_class=AdversarialSEMLearner \
  --config.learner_config.AdversarialSEMLearner.clipping_action=False \
  --config.learner_config.AdversarialSEMLearner.critic_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.critic_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.critic_l2_regularizer=1e-10 \
  --config.learner_config.AdversarialSEMLearner.critic_lr=0.001 \
  --config.learner_config.AdversarialSEMLearner.critic_regularizer=1e-06 \
  --config.learner_config.AdversarialSEMLearner.f_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.f_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.f_l2_regularizer=1e-08 \
  --config.learner_config.AdversarialSEMLearner.f_lr_multiplier=1.0 \
  --config.network_config.critic_layer_sizes=50,50 \
  --config.network_config.f_layer_sizes=150,150 \
  --config.problem_config.noise_level=$NOISE_LEVEL \
  --config.problem_config.task_name=bsuite_cartpole \
  --dataset_path="$DATASET_PATH"

# bsuite_catch
python run_deep_gmm.py \
  --config.learner_class=AdversarialSEMLearner \
  --config.learner_config.AdversarialSEMLearner.clipping_action=False \
  --config.learner_config.AdversarialSEMLearner.critic_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.critic_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.critic_l2_regularizer=1e-06 \
  --config.learner_config.AdversarialSEMLearner.critic_lr=1e-05 \
  --config.learner_config.AdversarialSEMLearner.critic_regularizer=1e-06 \
  --config.learner_config.AdversarialSEMLearner.f_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.f_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.f_l2_regularizer=1e-10 \
  --config.learner_config.AdversarialSEMLearner.f_lr_multiplier=1.0 \
  --config.network_config.critic_layer_sizes=50,50 \
  --config.network_config.f_layer_sizes=150,150 \
  --config.problem_config.noise_level=$NOISE_LEVEL \
  --config.problem_config.task_name=bsuite_catch \
  --dataset_path="$DATASET_PATH"

# bsuite_mountain_car
python run_deep_gmm.py \
  --config.learner_class=AdversarialSEMLearner \
  --config.learner_config.AdversarialSEMLearner.clipping_action=False \
  --config.learner_config.AdversarialSEMLearner.critic_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.critic_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.critic_l2_regularizer=1e-08 \
  --config.learner_config.AdversarialSEMLearner.critic_lr=0.0003 \
  --config.learner_config.AdversarialSEMLearner.critic_regularizer=1e-10 \
  --config.learner_config.AdversarialSEMLearner.f_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.f_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.f_l2_regularizer=0.0001 \
  --config.learner_config.AdversarialSEMLearner.f_lr_multiplier=1.0 \
  --config.network_config.critic_layer_sizes=50,50 \
  --config.network_config.f_layer_sizes=100,100 \
  --config.problem_config.noise_level=$NOISE_LEVEL \
  --config.problem_config.task_name=bsuite_mountain_car \
  --dataset_path="$DATASET_PATH"



## Pure offline datasets. DM Control Suite tasks.

# Valid values: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
NOISE_LEVEL=0.0

# dm_control_cartpole_swingup
python run_deep_gmm.py \
  --config.learner_class=AdversarialSEMLearner \
  --config.learner_config.AdversarialSEMLearner.clipping_action=True \
  --config.learner_config.AdversarialSEMLearner.critic_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.critic_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.critic_l2_regularizer=1e-10 \
  --config.learner_config.AdversarialSEMLearner.critic_lr=0.0001 \
  --config.learner_config.AdversarialSEMLearner.critic_regularizer=1e-08 \
  --config.learner_config.AdversarialSEMLearner.f_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.f_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.f_l2_regularizer=1e-10 \
  --config.learner_config.AdversarialSEMLearner.f_lr_multiplier=10.0 \
  --config.network_config.critic_layer_sizes=512,512,256 \
  --config.network_config.f_layer_sizes=768,768,384 \
  --config.problem_config.noise_level=$NOISE_LEVEL \
  --config.problem_config.task_name=dm_control_cartpole_swingup \
  --dataset_path="$DATASET_PATH"

# dm_control_cheetah_run
python run_deep_gmm.py \
  --config.learner_class=AdversarialSEMLearner \
  --config.learner_config.AdversarialSEMLearner.clipping_action=True \
  --config.learner_config.AdversarialSEMLearner.critic_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.critic_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.critic_l2_regularizer=1e-10 \
  --config.learner_config.AdversarialSEMLearner.critic_lr=0.0001 \
  --config.learner_config.AdversarialSEMLearner.critic_regularizer=1e-08 \
  --config.learner_config.AdversarialSEMLearner.f_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.f_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.f_l2_regularizer=1e-10 \
  --config.learner_config.AdversarialSEMLearner.f_lr_multiplier=10.0 \
  --config.network_config.critic_layer_sizes=512,512,256 \
  --config.network_config.f_layer_sizes=768,768,384 \
  --config.problem_config.noise_level=$NOISE_LEVEL \
  --config.problem_config.task_name=dm_control_cheetah_run \
  --dataset_path="$DATASET_PATH"

# dm_control_humanoid_run
python run_deep_gmm.py \
  --config.learner_class=AdversarialSEMLearner \
  --config.learner_config.AdversarialSEMLearner.clipping_action=True \
  --config.learner_config.AdversarialSEMLearner.critic_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.critic_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.critic_l2_regularizer=1e-06 \
  --config.learner_config.AdversarialSEMLearner.critic_lr=3e-05 \
  --config.learner_config.AdversarialSEMLearner.critic_regularizer=1e-08 \
  --config.learner_config.AdversarialSEMLearner.f_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.f_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.f_l2_regularizer=1e-10 \
  --config.learner_config.AdversarialSEMLearner.f_lr_multiplier=5.0 \
  --config.network_config.critic_layer_sizes=512,512,256 \
  --config.network_config.f_layer_sizes=1024,1024,512 \
  --config.problem_config.noise_level=$NOISE_LEVEL \
  --config.problem_config.task_name=dm_control_humanoid_run \
  --dataset_path="$DATASET_PATH"

# dm_control_walker_walk
python run_deep_gmm.py \
  --config.learner_class=AdversarialSEMLearner \
  --config.learner_config.AdversarialSEMLearner.clipping_action=True \
  --config.learner_config.AdversarialSEMLearner.critic_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.critic_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.critic_l2_regularizer=1e-10 \
  --config.learner_config.AdversarialSEMLearner.critic_lr=0.0001 \
  --config.learner_config.AdversarialSEMLearner.critic_regularizer=1e-08 \
  --config.learner_config.AdversarialSEMLearner.f_beta1=0.5 \
  --config.learner_config.AdversarialSEMLearner.f_beta2=0.9 \
  --config.learner_config.AdversarialSEMLearner.f_l2_regularizer=1e-10 \
  --config.learner_config.AdversarialSEMLearner.f_lr_multiplier=10.0 \
  --config.network_config.critic_layer_sizes=512,512,256 \
  --config.network_config.f_layer_sizes=768,768,384 \
  --config.problem_config.noise_level=$NOISE_LEVEL \
  --config.problem_config.task_name=dm_control_walker_walk \
  --dataset_path="$DATASET_PATH"
