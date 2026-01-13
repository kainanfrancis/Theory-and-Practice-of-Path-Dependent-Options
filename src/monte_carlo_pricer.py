import numpy as np
from typing import Callable


class MonteCarloPricer:
    """
    Generic Monte-Carlo pricer class that uses the risk-neutral pricing formula
    to value an option.

    Instance Attributes:
        S0 (float): Initial underlying asset price.
        T (float): Maturity (years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        n_sims (int): Number of simulations. Default: 1000.
        n_steps (int): Number of time steps in each simulation. Default: 252.

    Attributes: 
        dt (float): Time step size.
    """
    def __init__(
        self, S0: float, T: float, r: float, sigma: float,
        n_sims: int = 1000, n_steps: int = 252
    ):
        # Parameters
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_sims = n_sims
        self.n_steps = n_steps

        # Time step size
        self.dt: float = T / n_steps


    def _simulate_paths(self) -> np.ndarray:
        """
        Simulate GBM paths under the risk-neutral measure.

        Returns:
            np.ndarray: Array of shape (n_sims, n_steps+1) containing
            simulated paths.
        """
        # Generate standard normals
        Z = np.random.normal(size=(self.n_sims, self.n_steps))

        # Compute log-price increments
        increments = (
            (self.r - 0.5 * self.sigma**2) * self.dt
            + self.sigma * np.sqrt(self.dt) * Z
        )

        # Compute cumulative log-price paths
        log_paths = np.cumsum(increments, axis=1)

        # Insert initial log-price at time 0
        log_paths = np.hstack(
            [np.zeros((self.n_sims, 1)), log_paths]
        )

        # Convert to price paths
        return self.S0 * np.exp(log_paths)


    def price(
        self, H: Callable[[np.ndarray], np.ndarray], 
        return_std: bool=False
    ) -> float | tuple[float, float]:
        """
        Price a contingent claim using Monte Carlo simulation.

        Params:
            H (Callable[[np.ndarray], np.ndarray]: Payoff functional.
            return_std (bool): If true, also return the standard error
                               from the simulation

        Returns:
            float | tuple[float, float]: Monte Carlo price estimate. If
            return_std == True, also return standard error.
        """
        # Simulate price paths of the asset
        paths = self._simulate_paths()
        # Compute payoffs
        payoffs = H(paths)

        # Discount the payoffs
        discounted = np.exp(-self.r * self.T) * payoffs
        # Comoute the estimate as the mean of the discounted payoffs
        price = np.mean(discounted)

        # If return_std is true, compute and return standard error
        if return_std:
            std_error = np.std(discounted, ddof=1) / np.sqrt(self.n_sims)
            return price, std_error
        # Otherwise just return the estimator
        return price