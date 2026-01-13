import numpy as np
from scipy.stats import norm
from typing import Callable

from src.finite_difference_pricer_1D import FiniteDifferencePricer1D
from src.monte_carlo_pricer import MonteCarloPricer
from src.pricer import Pricer


class UpAndOutFD(FiniteDifferencePricer1D):
    """
    Prices up-and-out barrier options under the Black-Scholes model using 
    a finite difference scheme.

    Instance Attributes:
        S0 (float): Initial underlying asset price.
        h (Callable): Terminal payoff function.
        B (float): Barrier level.
        T (float): Maturity (years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        M (int): Number of spatial grid points. Default: 200.
        N (int): Number of time steps. Default: 200.

    Attributes:
        dx (float): Spatial step size.
        dt (float): Time step size.
        x (np.ndarray): Spatial grid.
        v (np.ndarray): Solution grid storing the option values v(t_n, x_i).
        tri (np.ndarray): Tridiagonal matrix defining the implicit FD scheme.
    """
    def __init__(
        self, S0: float, h: Callable[[np.ndarray], np.ndarray], 
        B: float, T: float, r: float, sigma: float, M: int=200, N: int=200
    ):
        self.h = h
        self.B = B
        # Use barrier level as x_max
        super().__init__(S0, T, r, sigma, B, M, N)

    def _terminal_condition(self):
        """Define terminal condition v(T,x) = h(x), 0 <= x <= B."""
        self.v[-1, :] = self.h(self.x)
        self.v[-1, self.x >= self.B] = 0.0

    def _apply_boundary_conditions(self, n):
        """Define boundary conditions v(t,0)=0, v(t,B)=0"""
        self.v[n, 0] = 0.0
        self.v[n, self.M] = 0.0


class UpAndOutMC(Pricer):
    """
    Prices up-and-out barrier options under the Black-Scholes model using 
    Monte-Carlo methods.

    Instance Attributes:
        S0 (float): Initial underlying asset price.
        h (Callable): Terminal payoff function.
        B (float): Barrier level.
        T (float): Maturity (years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        n_sims (int): Number of simulations. Default: 1000.
        n_steps (int): Number of time steps in each simulation. Default: 252.

    Attributes:
        mc (MonteCarloPricer): Base Monte Carlo pricer object for simulation.
    """
    def __init__(
        self, S0: float, h: Callable[float, float], B: float, T: float, r: float, 
        sigma: float, n_sims: int=1000, n_steps: int=252
    ):
        self.S0 = S0
        self.h = h
        self.B = B
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_sims=n_sims
        self.n_steps=n_steps

        # Instantiate base Monte Carlo pricer
        self.mc = MonteCarloPricer(
            S0=S0,
            T=T,
            r=r,
            sigma=sigma,
            n_sims=n_sims,
            n_steps=n_steps
        )


    def price(self, return_std=False) -> float:
        """Compute the up-and-out option price."""
        # Define the payoff function H
        def payoff(paths: np.ndarray) -> np.ndarray:
            # Knock-out condition: barrier hit at any time
            knocked_out = np.any(paths >= self.B, axis=1)

            # Get final values
            ST = paths[:, -1]
            # Compute payoffs
            payoffs = self.h(ST)
            # Set the payoff on knock-out to 0
            payoffs[knocked_out] = 0.0
            
            return payoffs

        # Compute and return the price 
        # (and standard error if return_std==True)
        return self.mc.price(payoff, return_std)    


def up_and_out_call_analytic(
    S0: float,
    K: float,
    B: float,
    T: float,
    t: float,
    r: float,
    sigma: float,
    q: float=0,
) -> float:
    """
    Implementation of the analytical up-and-out call price 
    (Formula in Wilmott [pp.~408-409]).

    Params:
        S0 (float): Initial underlying asset price.
        K (float): Strike price.
        B (float): Barrier level.
        T (float): Maturity (years).
        t (float): Valuation time (t <= T)
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        q (float): Continuously compounded dividend yield (or cost-of-carry/
        foreign exchange rate for commodity/currency options, respectively).

    Returns:
        float: Analytical price of the option in the Black-Scholes market.
    """
    # Check t <= T
    if t > T:
        raise ValueError("Valuation time t must not be greater than maturity T.")

    # Immediate knockout
    if S0 >= B:
        return 0.0

    # Time to maturity
    tau = T - t

    # Reused volatility/time coefficient
    sqrt_tau = np.sqrt(tau)
    vol_sqrt = sigma * sqrt_tau

    # Drift terms
    mu_p = (r - q + 0.5 * sigma**2) * tau
    mu_m = (r - q - 0.5 * sigma**2) * tau

    # Log terms
    log_SK  = np.log(S0 / K)
    log_SB = np.log(S0 / B)
    log_SKB2 = np.log(S0 * K / B**2)

    # d-terms
    d1 = (log_SK + mu_p) / vol_sqrt
    d2 = (log_SK + mu_m) / vol_sqrt

    d3 = (log_SB + mu_p) / vol_sqrt
    d4 = (log_SB + mu_m) / vol_sqrt

    d5 = (log_SB - mu_m) / vol_sqrt
    d6 = (log_SB - mu_p) / vol_sqrt

    d7 = (log_SKB2 - mu_m) / vol_sqrt
    d8 = (log_SKB2 - mu_p) / vol_sqrt

    # Barrier coefficients
    power = 2 * (r - q) / sigma**2
    a = (B / S0)**(-1 + power)
    b = (B / S0)**( 1 + power)

    # Discount factors
    disc_q = np.exp(-q * tau)
    disc_r = np.exp(-r * tau)

    # Final price
    price = (
        S0 * disc_q
        * (norm.cdf(d1) - norm.cdf(d3) - b * (norm.cdf(d6) - norm.cdf(d8)))
        -
        K * disc_r
        * (norm.cdf(d2) - norm.cdf(d4) - a * (norm.cdf(d5) - norm.cdf(d7)))
    )

    return price








