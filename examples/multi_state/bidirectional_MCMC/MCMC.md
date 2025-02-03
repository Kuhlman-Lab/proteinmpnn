This version uses a Markov Chain Monte Carlo sampling to optimize a pair of proteins via DNA mutations and can be implemented for phase 0, -1 & -2 overlapping proteins. Future updates will come for phase 1 & 2 (same strand). Ensure that the metro.flags file has strand=opposite otherwise it will not work.

The MCMC sampling routine uses a new config file, `metro.flags`. This configures the MCMC parameters, which are summarized below:
```
# Whether to use the same or opposite strand of DNA for encoding. Valid options are: {opposite, same}. Default is opposite.
# NOTE: 'strand=same' is not supported as of 2/3/2025.
strand=opposite 

# In base pairs, how many positions to shift the coding sequences relative to one another. Valid options are {0, 1, 2}. Default is 1.
shift=2

# Whether to use a temperature gradient for the Metropolis algorithm. Valid options are {True, False}. Default is False.
# If False, the 'temperature' value will be constant. 
# If True, it will take linear steps between 'gradient_start' and 'gradient_end'.
# Only used if 'metropolis=True'.
use_gradient=False

# Temperature for the Metropolis algorithm. Can be any positive value. Default is 0.007.
# Only used if 'use_gradient=False'.
temperature=0.007

# Start and end values for Metropolis temperature gradient. Can be any positive value. Default is 0.5 to 0.05.
# Only used if 'use_gradient=True'.
gradient_start=0.5
gradient_end=0.05

# Number of iterations for the MCMC sampler to run (one mutation per iteration). Recommended to be ~300x the protein length. Default is 30000.
num_mutations=30000

# Whether to use Metropolis algorithm to accept/reject points. # Valid options are {True, False}. Default is False.
# If True, then 'use_gradient' will be used to set temperature values. If False, then only points with lower (better) scores will be accepted.
metropolis=False
```