A transformer model to learn to sample Ising lattices at critical temperature.

The critical design choices are detailed below:

Data Sampling: We chose to work with the Wolff algorithm to obtain our data set that will use for the transformer model. 
This includes a burn-in phase to reduce correlation in the produced spins.

Underlying model: Standard autoregressive setup. We define the positional encodings with work on a flattened vector but should incorporate positions on a 2d grid, accounting for different spins using sine/cosines. We then have an embedding layer, perform attention with appropriate masking, and final linear layer back to +- space.

Evaluation: Ising lattice is sampled using the transformer, and the corresponding energy, magnetization and correlations are computed. These are then compared to the test set produced by the MCMC sampler. Correlations are measured by approximating E(sigma_i sigma_{i+r}) for varying r, and at critical temperature it is known that long-range correlations occur.

Results: The results showed the the model struggled to learn correlations and variance. I found a bug in the code where the BOS token was the same as spin-down, and so will see if changes this improved performance.
