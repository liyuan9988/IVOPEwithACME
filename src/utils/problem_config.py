"""ConfigDict of the problem configuration.

To use the pure off-policy dataset with a noise level `p`, set:
  use_near_policy_dataset = False
  prob_param.noise_level = target_policy_param.env_noise_level = p

To use the near-policy dataset, set
  use_near_policy_dataset = True
  prob_param.noise_level = p
  behavior_policy_param.env_noise_level = target_policy_param.env_noise_level = 0.0
"""

import ml_collections as collections


def get_problem_config():
  """OPE problem config."""
  problem_config = collections.ConfigDict({
      'task_name': 'bsuite_cartpole',  # Task name in the format of
                                       # 'bsuite_[bsuite_id]' or
                                       # 'dm_control_[takename]'
      'noise_level': 0.2,  # Environment noise level
      'near_policy_dataset': False,  # Use near-policy dataset if True,
                                     # otherwise use pure offline dataset.

#       # Config of the evaluation environment and dataset.
#       'prob_param': {
#           'noise_level': 0.2,  # Environment noise level in policy evaluation.
#           'run_id': 0  # From which run the pure offline dataset is generated.
#                        # Not effective if use_near_policy_dataset = True.
#       },

#       # Config of the target policy.
#       'target_policy_param': {
#           # How the policy is trained.
#           'env_noise_level': 0.2,  # Environment noise level in which the target
#                                    # policy is trained.
#           'run_id': 1,             # Id of the target policy training run.

#           # How the policy is run in evaluation.
#           'policy_noise_level': 0.1  # Action noise of the target policy.
#       },

#       # Config of the behavior policy for near-policy dataset.
#       # behavior_policy_param is not effective if
#       #   use_near_policy_dataset = False
#       'use_near_policy_dataset': False,  # Use near-policy dataset if True,
#                                          # otherwise use pure offline dataset.
#       'behavior_policy_param': {
#           # How the policy is trained.
#           'env_noise_level': 0.0,  # Environment noise level in which the
#                                    # behavior policy is trained.
#           'run_id': 1,             # Id of the behavior policy training run.

#           # How the policy is run in data generation.
#           'policy_noise_level': 0.3  # Action noise of the behavior policy.
#       },

      'discount': 0.99,  # Reward discount.
  })
  return problem_config
