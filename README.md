Introduction:

My own implementation of A2C, PPO, TRPO, AlphaZero, and PPO RND that can be applied to Connect4 and Atari games such as Pong and Seaquest. Results of PPO on Pong are in the out directory.

Initially, I tried to use a modified GAE calculation for the PPO/A2C/TRPO roll-out on Connect4, but it ran into issues. I think it is very hard to gauge how well PPO, A2C and TRPO are doing if it is just self play. Likely, it would be best to implement the AlphaZero style win-ratio. It would be probably more instructive to pit PPO/A2C or TRPO algorithms against a difficult opponent (like in Pong).

Soon the results of PPO RND will be implemented on Seaquest and included in the out directory.

Running the Experiments:

Experiments are mainly run using the SLURM scripts in Experiments/SLURM/sweepsrunpod.sh.

A user can specify what experiments should be run by editing directly the Experiments/configs/sweeps.py file. 