# A3RL

# Algorithm versioning:
RLPD stable version: v08
A3RL stable version: v10. Behavior should be closely (if not the same as v08).

# Changelogs
## Version 11
- Pretraining for 25%, and then A3RL
- Reduced alpha to really small. (0.04). We'll see what hpapens.
## Verion 10: (Jan 21, 9:15 PM)
- Density wasn't switched. see line 165-190: https://github.com/shlee94/Off2OnRL/blob/main/rlkit/rlkit/torch/sac/ours.py. Previous change was reverted.
## Version 9: (Jan 21, 1:01 PM)
- Density network seems like it was switched. Should be correct now.
- Added logging priority buffer entropy
## Version 8: (Jan 20, 10:11PM)
- Density network takes UTD_ratio gradient steps with separate batch
- Priority Buffer has both offline and online. Only the last minibatch has its priority updated (since actor is only run on this minibatch). An extra run of density network through this (mixed offline and online) minibatch is required (can't reuse density values from the training of the density network itself).
- Advantage_LCB has hyperparameter $\beta = -0.2$, which corresponds to roughly 25%. Advantage LCB of the minibatch above is normalized to Z-score against the advantage LCB of an extra offline minibatch; so that $\lambda$ can be independent of environment.
- Importance sampling weights are NOT normalized to have max = 1, but rather so that their sum = UTD_ratio, so that it's similar to the uniform case. Previous normalizations to have max = 1 yield mean_weight ~ dubious amounts (too much/too little)


