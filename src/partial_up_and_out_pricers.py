import numpy as np
from typing import Callable

from src.finite_difference_pricer_1D import FiniteDifferencePricer1D
from src.monte_carlo_pricer import MonteCarloPricer
from src.pricer import Pricer


class PartialUpAndOutFD(Pricer):
    """
    Finite difference pricer for an active up-and-out partial barrier option.

    Instance Attributes:
        S0 (float): Initial underlying asset price.
        h (Callable): Terminal payoff function.
        B (float): Barrier level.
        Pi (list[(float,float)]: Collection of time intervals on which the
                                 barrier is active.
        T (float): Maturity (years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        x_max (float): Upper bound for x values.
        M (int): Number of spatial grid points.
        N (int): Number of time steps PER INTERVAL.

    Attributes:
        dx (float): Spatial step size.
        dt (float): Time step size.
        x (np.ndarray): Spatial grid.
        v (np.ndarray): Solution grid storing the option values v(t_n, x_i).
        tri (np.ndarray): Tridiagonal matrix defining the implicit FD scheme.
    """
    def __init__(
        self, S0: float, h: Callable[[np.ndarray], np.ndarray], B: float,
        Pi: list[tuple[float, float]], T: float, r: float, sigma: float,
        x_max: float, M: int, N: int
    ):
        # Model Parameters
        self.S0 = S0
        self.h = h
        self.B = B
        self.Pi = Pi
        self.T = T
        self.r = r
        self.sigma = sigma

        # Pricer Parameters
        self.x_max = x_max
        self.M = M
        self.N = N

        # Get partition times
        self.times = self._build_partition()

    def _build_partition(self) -> list[float]:
        """
        Build the Pi-induced time partition: 0 = t0 < t1 < ... < t_{2N} = T.
        """
        points = {0.0, self.T}
        for a, b in self.Pi:
            points.add(a)
            points.add(b)
        return sorted(points)

    def interval_is_active(
        self,
        t_start: float,
        t_end: float,
    ) -> bool:
        """
        Check whether [t_start, t_end) intersects Π.
        """
        return any(
            not (t_end <= a or t_start >= b)
            for a, b in self.Pi
        )

    def _terminal_condition(
        self, pricer: FiniteDifferencePricer1D, 
        v_next: np.ndarray | None, x_next: np.ndarray | None
    ) -> None:
        """Define terminal condition for subinterval."""
        if v_next is None:
            # Final maturity: v(T,x)=h(x)
            pricer.set_terminal_condition(self.h(pricer.x))
        else:
            # Glue from next interval
            pricer.set_terminal_condition(
                np.interp(pricer.x, x_next, v_next)
            )
        
    def _apply_active_barrier_bc(
        self, pricer: FiniteDifferencePricer1D
    ) -> None:
        """
        Apply boundary conditions when the barrier is active.
        """
        def bc(n: int) -> None:
            # Absorbing boundary at S = 0
            pricer.v[n, 0] = 0.0

            # Barrier at S = B
            pricer.v[n, -1] = 0.0

        pricer._apply_boundary_conditions = bc

    def _apply_inactive_barrier_bc(
        self,
        pricer: FiniteDifferencePricer1D,
    ) -> None:
        """
        Apply boundary conditions when the barrier is inactive.
        """
        def bc(n: int) -> None:
            # Absorbing boundary at S = 0
            pricer.v[n, 0] = 0.0

            # Far-field Neumann boundary
            pricer.v[n, -1] = pricer.v[n, -2]

        pricer._apply_boundary_conditions = bc

    def price(self) -> float:
        """Compute option price at time t=0."""
        v_next: np.ndarray | None = None
        x_next: np.ndarray | None = None

        # Backward induction over time intervals
        for j in reversed(range(len(self.times)-1)):
            # Get start and end times for the interval
            t_start = self.times[j]
            t_end = self.times[j+1]

            # Start to end time of the interval
            dt = t_end - t_start

            # Determine whether interval is active
            active = self.interval_is_active(t_start, t_end)

            # Spatial domain
            x_max = self.B if active else self.x_max

            # Build FD solver on the interval
            pricer = FiniteDifferencePricer1D(
                S0 = self.S0,
                T=dt,
                r=self.r,
                sigma=self.sigma,
                x_max=x_max,
                M=self.M,
                N=self.N
            )

            # Terminal condition
            self._terminal_condition(pricer, v_next, x_next)

            # Boundary conditions
            if active:
                self._apply_active_barrier_bc(pricer)
            else:
                self._apply_inactive_barrier_bc(pricer)

            # Solve PDE on this interval
            v_next = pricer.price(return_grid=True)
            x_next = pricer.x

        # Final interpolation at S0
        return float(np.interp(self.S0, x_next, v_next))


class PartialUpAndOutMC(Pricer):
    """
    Monte Carlo pricer for active partial up-and-out barrier options
    under the Black–Scholes model.

    Instance Attributes:
        S0 (float): Initial underlying asset price.
        h (Callable): Terminal payoff function.
        B (float): Barrier level.
        Pi (list[(float,float)]: Collection of time intervals on which the
                                 barrier is active.
        T (float): Maturity (years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        n_sims (int): Number of simulations. Default: 1000.
        n_steps (int): Number of time steps in each simulation. Default: 252.

    Attributes: 
        dt (float): Time step size.
    """
    def __init__(
        self, S0: float, h: Callable[[np.ndarray], np.ndarray], B: float,
        Pi: list[tuple[float, float]], T: float, r: float, sigma: float,
        n_sims: int = 1000, n_steps: int = 252
    ):
        # Model parameters
        self.S0 = S0
        self.h = h
        self.B = B
        self.Pi = Pi
        self.T = T
        self.r = r
        self.sigma = sigma

        # Simulation parameters
        self.n_sims = n_sims
        self.n_steps = n_steps

        # Base Monte Carlo engine
        self.mc = MonteCarloPricer(
            S0=S0,
            T=T,
            r=r,
            sigma=sigma,
            n_sims=n_sims,
            n_steps=n_steps,
        )

        # Time grid
        self.dt = T / n_steps
        self.times = np.linspace(0.0, T, n_steps + 1)

        # Precompute which time indices are in Pi
        self.active_idx = self._active_time_indices()

    def _active_time_indices(self) -> np.ndarray:
        """
        Return a boolean mask indicating which time steps
        lie in Pi.
        """
        mask = np.zeros(self.n_steps + 1, dtype=bool)

        for j, t in enumerate(self.times):
            for t0, t1 in self.Pi:
                if t0 <= t < t1:
                    mask[j] = True
                    break

        return mask

    def price(self, return_std: bool = False) -> float | tuple[float, float]:
        """Price the option"""    
        def payoff(paths: np.ndarray) -> np.ndarray:
            """
            Payoff function.
            """
            # Check barrier condition only during active times in Pi
            active_paths = paths[:, self.active_idx]
            barrier_hit = np.any(active_paths >= self.B, axis=1)

            # Terminal payoff
            ST = paths[:, -1]
            base_payoff = self.h(ST)

            # Payoff only if barrier was not hit
            payoffs = np.where(barrier_hit, 0.0, base_payoff)

            return payoffs

        # Price the option
        return self.mc.price(payoff, return_std)



        