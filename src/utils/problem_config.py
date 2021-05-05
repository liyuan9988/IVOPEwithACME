"""ConfigDict of the problem configuration."""

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
      'discount': 0.99,  # Reward discount.
  })
  return problem_config
