import numpy as np
from typing import Callable

from src.finite_difference_pricer_2D import FiniteDifferencePricer2D
from src.monte_carlo_pricer import MonteCarloPricer
from src.pricer import Pricer


class ModifiedUpAndOutFD(FiniteDifferencePricer2D):
    """
    Modified up-and-out barrier option with occupation time.
    Uses characteristic-line discretisation.

    Instance Attributes:
        S0 (float): Initial underlying asset price.
        h (Callable): Terminal payoff function.
        Phi (Callable): Phase out function.
        B (float): Barrier level.
        T (float): Maturity (years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        x_max (float): Upper bound for x values.
        M (int): Number of spatial grid points.
        K (int): Number of grid points in tau.
        N (int): Number of time steps.
        continuous (bool): Uses continuous occupation time if True,
                           otherwise uses accumulated occupation time.
                           Default: False.

    Attributes:
        dx (float): Spatial step size.
        dtau (float): Step size in tau.
        dt (float): Time step size.
        x (np.ndarray): Spatial grid.
        tau (np.ndarray): tau grid.
        v (np.ndarray): Solution grid storing the option values v(t_n, x_i).
        tri (np.ndarray): Tridiagonal matrix defining the implicit FD scheme.
    """

    def __init__(
        self,
        S0: float,
        h: Callable[[np.ndarray], np.ndarray],
        Phi: Callable[[np.ndarray], np.ndarray],
        B: float,
        T: float,
        r: float,
        sigma: float,
        x_max: float,
        M: int,
        K: int,
        N: int,
        continuous: bool = False,
    ):
        # Modified up-and-out specific attributes
        self.h = h
        self.Phi = Phi
        self.B = B
        self.continuous = continuous

        # Monte-Carlo initialisation
        super().__init__(
            S0=S0,
            T=T,
            r=r,
            sigma=sigma,
            x_max=x_max,
            tau_max=T,
            M=M,
            K=K,
            N=N,
        )

    def _terminal_condition(self):
        """Define terminal condition v(T,x,tau) = h(x)Phi(tau)."""
        for i in range(self.M + 1):
            self.v[-1, i, :] = self.h(self.x[i]) * self.Phi(self.tau)

    def _characteristic_tau(self, i: int, tau_l: float) -> float:
        """
        Characteristic:
          below barrier: tau constant.
          above barrier: tau increases at unit speed.
        """
        if self.x[i] < self.B:
            return tau_l
        else:
            return tau_l + self.dt

    def _apply_reset_condition(self, n: int):
        """Apply reset condition if needed."""
        # If not continuous occupation time, reset not needed
        if not self.continuous:
            return

        # Otherwise apply reset condition
        for i in range(self.M + 1):
            if self.x[i] < self.B:
                self.v[n, i, :] = self.v[n, i, 0]

    def _apply_boundary_conditions(self, n: int, l: int):
        """Apply boundary conditions v(t,0,tau)=0, far field boundary far out."""
        # S = 0
        self.v[n, 0, l] = 0.0
        # far-field (Neumann)
        self.v[n, self.M, l] = self.v[n, self.M - 1, l]


class ModifiedUpAndOutMC(Pricer):
    """
    Monte Carlo pricer for modified up-and-out options
    with occupation-time-based phase-out.

    Instance Attributes:
        S0 (float): Initial underlying asset price.
        h (Callable): Terminal payoff function.
        Phi (Callable): Phase out function.
        B (float): Barrier level.
        T (float): Maturity (years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        n_sims (int): Number of simulations. Default: 1000.
        n_steps (int): Number of time steps in each simulation. Default: 252.
        continuous (bool): Uses continuous occupation time if True,
                           otherwise uses accumulated occupation time.
                           Default: False.

    Attributes: 
        dt (float): Time step size.
    """
    def __init__(
        self,
        S0: float,
        h: Callable[[np.ndarray], np.ndarray],
        Phi: Callable[[np.ndarray], np.ndarray],
        B: float,
        T: float,
        r: float,
        sigma: float,
        n_sims: int = 1000,
        n_steps: int = 252,
        continuous: bool = False,
    ):
        # Model parameters
        self.S0 = S0
        self.h = h
        self.Phi = Phi
        self.B = B
        self.T = T
        self.r = r
        self.sigma = sigma

        # Simulation parameters
        self.n_sims = n_sims
        self.n_steps = n_steps
        self.continuous = continuous

        # Initialise Monte-Carlo pricers
        self.mc = MonteCarloPricer(
            S0=S0,
            T=T,
            r=r,
            sigma=sigma,
            n_sims=n_sims,
            n_steps=n_steps,
        )

        # Time-step
        self.dt = T / n_steps

    def price(self, return_std: bool = False):
        """
        Compute the modified up-and-out option price.
        """
        def payoff(paths: np.ndarray) -> np.ndarray:
            """Payoff function."""
            # Terminal prices
            ST = paths[:, -1]

            # Occupation time
            if not self.continuous:
                # accumulated occupation time
                indicator = paths[:, 1:] >= self.B
                tau = indicator.sum(axis=1) * self.dt
            else:
                # continuous occupation time
                tau = np.zeros(self.n_sims)
                for i in range(self.n_sims):
                    count = 0
                    for k in range(self.n_steps):
                        if paths[i, -(k + 1)] >= self.B:
                            count += 1
                        else:
                            break
                    tau[i] = count * self.dt

            # Payoff with phase-out
            payoffs = self.h(ST) * self.Phi(tau)
            return payoffs

        # Price the option
        return self.mc.price(payoff, return_std)
        