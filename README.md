# A3RL

# Algorithm versioning:
RLPD stable version: v08
A3RL stable version: v08

# Changelogs
## Version 8: (Jan 20, 10:11PM)
- Density network takes UTD_ratio gradient steps with separate batch
- Priority Buffer has both offline and online. Only the last minibatch has its priority updated (since actor is only run on this minibatch). An extra run of density network through this (mixed offline and online) minibatch is required (can't reuse density values from the training of the density network itself).
- Advantage_LCB has hyperparameter $\beta = -0.2$, which corresponds to roughly 25%. Advantage LCB of the minibatch above is normalized to Z-score against the advantage LCB of an extra offline minibatch; so that $\lambda$ can be independent of environment.
- Importance sampling weights are NOT normalized to have max = 1, but rather so that their sum = UTD_ratio, so that it's similar to the uniform case. Previous normalizations to have max = 1 yield mean_weight ~ dubious amounts (too much/too little)


