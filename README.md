# Black box optimization methods
We compare two "black box" or "derivative-free" optimization methods: the *cross-entropy method* and the *covariance matrix adaptation evolutionary strategy* (CMA-ES).

This class of optimizations do not assume smoothness or convexity of the objective function. Instead, it places a proposal distribution on the domain
of the function and updates the parameters of the distribution at each step. CMA-ES is designed to work with the multivariate normal.
For comparison, we use a multivariate normal proposal distribution for the cross-entropy method as well, calling it CEMVN.

Much of the implementation of CEMVN and CMA-ES is drawn from *Algorithms for Optimization* by Kochenderfer and Wheeler (https://algorithmsbook.com/optimization/),
but the code to generate the plots is original.

## Performance on the 20-dimensional Michalewicz test function
![michalewicz](michalewicz_compare.png?raw=true)
