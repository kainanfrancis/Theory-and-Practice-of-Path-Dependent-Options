import pandas as pd


def comparison(
    data: list[list[str, str, float]], analytic_price: float, errors=False
) -> pd.DataFrame:
    """Convert data into Dataframe, possibly with errors computed."""
    # Put data into a list of lists, where each list is a row of data with errors computed 
    rows = []
    for method, params, price in data:
        if errors:
            # Compute errors
            abs_err = abs(price - analytic_price)
            rel_err = abs_err / analytic_price
            rows.append([method, params, price, abs_err, rel_err])
        else:
            rows.append([method, params, price])
    # Convert data to dataframe
    return pd.DataFrame(
        rows,
        columns=["Method", "Parameters", "Price", "Absolute Error", "Relative Error"]
    )