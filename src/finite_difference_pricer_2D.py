import numpy as np
from src.pricer import Pricer


class FiniteDifferencePricer2D(Pricer):
    """
    Base class for two-dimensional finite-difference pricing under Black–Scholes
    with occupation time as an additional state variable. Solves prices of the form
    V(t) = v(t, x, tau) using an implicit method with characteristic-line.

    Subclasses should override:
        _terminal_condition: specifies payoff at maurity.
        _apply_boundary_condition: specifies boundary conditions.
        _characteristic_tau: defines tau evolution along characteristics.
        _apply_reset_condition: optional reset for continuous occupation times.

    Instance Attributes:
        S0 (float): Initial underlying asset price.
        T (float): Maturity (years).
        r (float): Risk-free interest rate.
        sigma (float): Volatility.
        x_max (float): Upper bound for x values.
        tau_max (float): Upper bound for tau values.
        M (int): Number of spatial grid points.
        K (int): Number of grid points in tau.
        N (int): Number of time steps.

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
        T: float,
        r: float,
        sigma: float,
        x_max: float,
        tau_max: float,
        M: int,
        K: int,
        N: int,
    ):
        # Model Parameters
        self.S0 = S0
        self.T = T
        self.r = r
        self.sigma = sigma

        # Grid sizes
        self.M = M
        self.K = K    
        self.N = N    

        # Grid spacings
        self.dx = x_max / M
        self.dtau = tau_max / K
        self.dt = T / N

        # Spatial grids
        self.x = np.linspace(0.0, x_max, M + 1)
        self.tau = np.linspace(0.0, tau_max, K + 1)

        # Solution grid
        # v[n, i, l] = v(t_n, x_i, tau_l)
        self.v = np.zeros((N + 1, M + 1, K + 1))

        # Precompute A_i, B_i, C_i
        self._setup_coefficients()

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
    
    def _terminal_condition(self):
        """Apply terminal payoff."""
        raise NotImplementedError

    def _apply_boundary_conditions(self, n: int, l: int):
        """Apply boundary conditions."""
        raise NotImplementedError

    def _apply_reset_condition(n):
        """Apply reset condition."""
        raise NotImplementedError

    def price(self) -> float:
        """Compute the option price at t=0, S(0)=S0 (and tau=0)."""
        # Set terminal payoff
        self._terminal_condition()

        # Backward time-stepping
        for n in range(self.N - 1, -1, -1):

            # Characteristic transport in tau
            v_star = np.zeros_like(self.v[n + 1])

            for i in range(self.M + 1):
                for l in range(self.K + 1):
                    tau_shift = self._characteristic_tau(i, self.tau[l])
                    v_star[i, l] = self._interp_tau(
                        self.v[n + 1, i, :], tau_shift
                    )

            # Implicit BS solve in x (slice-by-slice in tau)
            for l in range(self.K + 1):
                rhs = v_star[1:self.M, l]
                self.v[n, 1:self.M, l] = np.linalg.solve(self.tri, rhs)
                self._apply_boundary_conditions(n, l)

            # Reset condition for continuous occupation time
            self._apply_reset_condition(n)

        # price uses tau = 0 slice
        return np.interp(self.S0, self.x, self.v[0, :, 0])

    def _characteristic_tau(self, i: int, tau_l: float) -> float:
        """
        Return tau(t_{n+1}) along the characteristic
        traced back from (t_n, tau_l).
        """
        raise NotImplementedError

    def _interp_tau(self, values: np.ndarray, tau_val: float) -> float:
        """
        Linear interpolation in tau.
        """
        if tau_val <= self.tau[0]:
            return values[0]
        if tau_val >= self.tau[-1]:
            return values[-1]
        return np.interp(tau_val, self.tau, values)
        