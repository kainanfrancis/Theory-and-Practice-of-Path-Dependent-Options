import numpy as np
from typing import Callable
from src.pricer import Pricer


class FiniteDifferencePricer1D(Pricer):
    """
    Base class for implicit finite-difference pricing under Black–Scholes in 1 dimension,
    i.e. for V(t)=v(t,x).

    Subclasses should override:
        _terminal_condition: specifies the payoff at maturity.
        _apply_boundary_condition: specifies boundary conditions.

    Instance Attributes:
        S0 (float): Initial underlying asset price.
        T (float): Maturity (years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        x_max (float): Upper bound for x values.
        M (int): Number of spatial grid points.
        N (int): Number of time steps.

    Attributes:
        dx (float): Spatial step size.
        dt (float): Time step size.
        x (np.ndarray): Spatial grid.
        v (np.ndarray): Solution grid storing the option values v(t_n, x_i).
        tri (np.ndarray): Tridiagonal matrix defining the implicit FD scheme.
    """
    def __init__(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        x_max: float,
        M: int,
        N: int,
    ):
        # Model Parameters
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma

        # Grid sizes
        self.M = M
        self.N = N

        # Grid spacings
        self.dx = x_max / M
        self.dt = T / N

        # Spatial grid
        self.x = np.linspace(0.0, x_max, M + 1)
        # Solution grid
        # v[n, i] = v(t_n, x_i)
        self.v = np.zeros((N + 1, M + 1))

        # Precompute A_i, B_i, C_i
        self._setup_coefficients()

        # Flag for terminal condition
        self._terminal_condition_set = False

    def _setup_coefficients(self) -> None:
        """Precompute A_i, B_i, C_i."""
        # Interior spatial indices
        i = np.arange(1, self.M)
        xi = self.x[i]

        # Get step sizes
        dx, dt = self.dx, self.dt

        # Compute coefficients
        self.Ai = self._Ai(xi, dt, dx)
        self.Bi = self._Bi(xi, dt, dx)
        self.Ci = self._Ci(xi, dt, dx)

        # Assemble matrix
        self.tri = np.zeros((self.M - 1, self.M - 1))
        np.fill_diagonal(self.tri, self.Bi)
        np.fill_diagonal(self.tri[1:], self.Ai[1:])
        np.fill_diagonal(self.tri[:, 1:], self.Ci[:-1])

    def _Ai(
        self, xi: np.ndarray, dt: float, dx: float
    ) -> np.ndarray:
        """Compute A_i."""
        return -dt * (0.5 * self.sigma**2 * xi**2 / dx**2 - self.r * xi / (2 * dx))

    def _Bi(
        self, xi: np.ndarray, dt: float, dx: float
    ) -> np.ndarray:
        """Compute B_i."""
        return 1 + dt * (self.sigma**2 * xi**2 / dx**2 + self.r)
    
    def _Ci(
        self, xi: np.ndarray, dt: float, dx: float
    ) -> np.ndarray:
        """Compute C_i."""
        return -dt * (0.5 * self.sigma**2 * xi**2 / dx**2 + self.r * xi / (2 * dx))

    def _terminal_condition(self) -> None:
        """Apply terminal payoff."""
        raise NotImplementedError

    def set_terminal_condition(self, values: np.ndarray) -> None:
        """Method to set terminal conditions externally."""
        if values.shape != self.v[-1,:].shape:
            raise ValueError("Terminal condition has incorrect shape.")
        self.v[-1,:] = values
        self._terminal_condition_set = True

    def _apply_boundary_conditions(self, n: int) -> None:
        """Apply boundary conditions."""
        raise NotImplementedError

    def price(self, return_grid: bool=False) -> float:
        """Compute the option price at t=0, S(0)=S0."""
        # Only apply terminal condition if not already set
        if not self._terminal_condition_set:
            # Set terminal payoff at maturity T
            self._terminal_condition()

        # Backward time-stepping
        for n in range(self.N - 1, -1, -1):
            # Right hand side from next time step
            rhs = self.v[n + 1, 1:self.M]

            # Solve system for interior points
            self.v[n, 1:self.M] = np.linalg.solve(self.tri, rhs)
            # Apply the boundary conditions
            self._apply_boundary_conditions(n)

        # Return grid if return_grid==True
        if return_grid:
            return self.v[0].copy()
        
        # Interpolate the solution at S0
        return np.interp(self.S0, self.x, self.v[0])
