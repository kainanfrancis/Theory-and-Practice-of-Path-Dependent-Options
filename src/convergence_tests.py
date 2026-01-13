import time
import inspect
import numpy as np
import matplotlib.pyplot as plt
from typing import Iterable
from tqdm import tqdm

from src.pricer import Pricer


def fd_convergence(
    pricer_class: type[Pricer], params: dict, Ms: Iterable[int]
) -> tuple[Iterable[int], list[float], list[float]]:
    """
    Perform FD convergence study for FD pricer.

    Params:
        pricer_class (type[Pricer]): FD pricer class.
        params (dict): Disctionary of parameters (excluding M, N).
        Ms (Iterable[int]): Sequence of grid sizes to test (M=N).

    Returns:
        Ms (Iterable[int]): Sequence of grid sizes used (M=N).
        prices (list[float]): FD prices corresponding to Ms.
        times (list[float]): Time taken (in seconds) for each pricing run.
    """
    # List to store prices and times
    prices: list[float] = []
    times: list[float] = []

    use_K = "K" in (
         inspect.signature(pricer_class.__init__).parameters   
    )

    # Loop over M values
    for M in tqdm(Ms):
        if use_K:
            pricer = pricer_class(
                N=M,
                M=M,
                K=M,
                **params
            )
        else:
            pricer = pricer_class(
                N=M,
                M=M,
                **params
            )
        # Price the option and time it
        start = time.perf_counter()
        price = pricer.price()
        end = time.perf_counter()
        # Append price to prices list
        prices.append(price)
        times.append(end - start)

    return Ms, prices, times


def mc_convergence(
    pricer_class: type[Pricer], params: dict, Ns: Iterable[int], n_steps: int=252
) -> tuple[np.ndarray, list[float], list[float]]:
    """
    Perform MC convergence study for MC pricer.

    Params:
        pricer_class (type[Pricer]): MC pricer class.
        params (dict): Dictionary of parameters.
        Ns (Iterable[int]): Sequence of Monte Carlo simulation sizes to test.
        n_steps (int): Number of time steps in each simulation. Default: 252.

    Returns:
        Ns (Iterable[int]): Sequence of simulation sizes used.
        mc_prices (list[float]): MC prices corresponding to Ns.
        mc_errs (list[float]): MC standard errors corresponding to Ns.
        times (list[float]): Time taken (in seconds) for each pricing run.
    """
    
    # Lists to store prices and standard errors
    prices: list[float] = []
    std_errs: list[float] = []
    times: list[float] = []

    # Loop over N values
    for N in tqdm(Ns):
        pricer = pricer_class(
            n_sims=N,
            n_steps=n_steps,
            **params
        )
        # Compute price and standard error and time it
        start = time.perf_counter()
        price, stderr = pricer.price(return_std=True)
        end = time.perf_counter()
        # Add price and standard error to lists
        prices.append(price)
        std_errs.append(stderr)
        times.append(end - start)

    return Ns, prices, std_errs, times


def plot_fd_convergence_results(
    size: np.ndarray,
    prices: list[float],
    times: list[float],
    title: str,
    savefile: str,
    analytic_price: float | None = None
) -> None:
    """
    Plots the finite difference convergence test results.

    Params:
        size (np.ndarray): Array of grid/simulation sizes.
        prices (list[float]): List of numerical prices corresponding to
                              the size array.
        times (list[float]): List of runtimes corresponding to the size array.
        title (str): Title for the plot.
        savefile (str): Name of file to save the plot to.
        analytic_price (float | None): Analytic price. If none, analytic price
                                       is not plotted.
    """
    # Create figure and primary axis
    fig, ax_price = plt.subplots()
    
    # Price axis (left)
    ax_price.plot(size, prices, marker="o", label="FD price")
    if analytic_price:
        ax_price.axhline(
            analytic_price,
            color="black",
            linestyle="--",
            label="Analytic"
        )
    ax_price.set_xlabel("Grid size M = N")
    ax_price.set_ylabel("Option price")
    
    # Time axis (right)
    ax_time = ax_price.twinx()
    ax_time.plot(
        size,
        times,
        marker="s",
        linestyle=":",
        color="tab:red",
        label="Runtime"
    )
    ax_time.set_ylabel("Runtime (seconds)")
    
    # Combine legends from both axes
    lines_price, labels_price = ax_price.get_legend_handles_labels()
    lines_time, labels_time = ax_time.get_legend_handles_labels()
    ax_price.legend(
        lines_price + lines_time,
        labels_price + labels_time,
        loc="best"
    )
    
    # Title and save
    plt.title(title)
    plt.savefig(savefile)
    plt.show()


def plot_mc_convergence_results(
    size: np.ndarray,
    prices: np.ndarray,
    errs: np.ndarray,
    times: np.ndarray,
    title: str,
    savefile: str,
    analytic_price: float | None = None
) -> None:
    """
    Plots the Monte Carlo convergence test results.

    Params:
        size (np.ndarray): Array of grid/simulation sizes.
        prices (np.ndarray): List of numerical prices corresponding to
                              the size array.
        errs (np.ndarray): List of errors corresponding to the size array.
        times (np.ndarray): List of runtimes corresponding to the size array.
        title (str): Title for the plot.
        savefile (str): Name of file to save the plot to.
        analytic_price (float | None): Analytic price. If none, analytic price
                                       is not plotted.
    """
    # Get lower and upper regions
    upper = prices + errs
    lower = prices - errs
    
    # Create figure and primary axis
    fig, ax_price = plt.subplots()
    
    # Price axis (left)
    ax_price.plot(size, prices, marker="o", label="MC estimate")
    ax_price.fill_between(size, lower, upper, alpha=0.3, label="+/-1 SE band")

    if analytic_price is not None:
        ax_price.axhline(
            analytic_price,
            color="black",
            linestyle="--",
            label="Analytic"
        )

    ax_price.set_xscale("log")
    ax_price.set_xlabel("Number of paths")
    ax_price.set_ylabel("Option price")
    
    # Time axis (right)
    ax_time = ax_price.twinx()
    ax_time.plot(
        size,
        times,
        marker="s",
        linestyle=":",
        color="tab:red",
        label="Runtime"
    )
    ax_time.set_ylabel("Runtime (seconds)")
    
    # Combine legends from both axes
    lines_price, labels_price = ax_price.get_legend_handles_labels()
    lines_time, labels_time = ax_time.get_legend_handles_labels()
    ax_price.legend(
        lines_price + lines_time,
        labels_price + labels_time,
        loc="best"
    )
    
    # Title and save
    plt.title(title)
    plt.savefig(savefile)
    plt.show()
