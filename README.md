An implementation of A2C, PPO, TRPO and AlphaZero, originally for the connect 4 environment, but now more generally for Pong and Seaquest. I've tried to follow very closely the layout of the AC-conjecture REPO. The PPO, TRPO and A2C implementation are pretty much a direct implementation of those REPOs.

I am interested in seeing the role of LR scheduling in learning soon! I took the main Connect4 environment utils from Maxim's book, as well as the implementation of MCTS and AlphaZero. Again though, I tried to follow the main flow of the AC-conjecture REPO.

Initially, I tried to use a modified GAE calculation for the PPO/A2C/TRPO roll-out, but it ran into issues. I think it is very hard to gauge how well PPO, A2C and TRPO are doing if it is just self play. Likely, it would be best to implement the AlphaZero style win-ratio. It would be probably more instructive to pit PPO/A2C or TRPO algorithms against a difficult opponent (like in Pong).

Results of PPO on Pong are in the out-directory.