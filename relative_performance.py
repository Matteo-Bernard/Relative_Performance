import pandas as pd
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import tempfile

def beta(asset: pd.Series, market: pd.Series, way: Literal['+', '-', 'all']):
    """
    Description:
    Beta is a measure of a financial instrument's sensitivity to market movements. A beta of 1 indicates the asset tends
    to move in line with the market, a beta greater than 1 suggests higher volatility, and a beta less than 1 indicates
    lower volatility compared to the market.

    Parameters:
    - asset (pd.Series): Time series data representing the returns of the asset.
    - market (pd.Series): Time series data representing the returns of the market.
    - way (Literal['+', '-', 'all']): Specifies which type of data points should be considered for the beta calculation:
        - '+' (positive): Only considers periods where the asset's returns are positive. This is useful for measuring
          the beta when the asset is performing well.
        - '-' (negative): Only considers periods where the asset's returns are negative. This is useful for measuring
          the beta when the asset is underperforming.
        - 'all': Considers all periods without any filtering, giving the traditional beta measurement.

    Returns:
    - float: Beta coefficient, which measures the asset's sensitivity to market movements based on the specified filter.
    """
    df = pd.concat([asset, market], axis=1).dropna().pct_change().dropna()

    if way == '+':
        df = df[df.iloc[:, 0] > 0] 
    elif way == '-':
        df = df[df.iloc[:, 0] < 0]
    elif way == 'all':
        pass

    covariance = df.cov().iloc[1, 0]
    market_variance = df.iloc[:, 1].var()
    return covariance / market_variance

def alpha_jensen(asset, market, riskfree):
    """
    Description:
    Jensen's alpha is a risk-adjusted performance metric that represents the excess return of an asset over its expected
    return, given its beta and the expected return of the market. It is calculated as the difference between the actual
    return and the expected return based on the Capital Asset Pricing Model (CAPM).

    Parameters:
    - asset (pd.Series): Time series data representing the returns of the asset.
    - market (pd.Series): Time series data representing the returns of the market.
    - riskfree (float): Risk-free rate of return, typically the yield of a government bond.

    Returns:
    - float: Jensen's alpha, which measures the excess return of the asset over its expected return.
    """

    asset_beta = beta(asset, market, 'all')
    asset_return = asset.iloc[-1] / asset.iloc[0] - 1
    market_return = market.iloc[-1] / market.iloc[0] - 1
    risk_free_return = (1+riskfree)**(len(asset)/252) - 1

    return asset_return - (risk_free_return + asset_beta * (market_return - risk_free_return))

def relative_performance(history, market, risk_free, window_alpha=25, window_beta=25, lookback_steps=[ 0,  2,  4,  6,  8, 10], show=True):
    """
    Plots the relative performance of different assets based on Jensen's Alpha and Beta over varying lookback periods.

    Parameters:
    ----------
    history : pd.DataFrame
        DataFrame containing historical returns of different assets (columns represent tickers).
    market : pd.Series
        Series containing historical returns of the market index.
    risk_free : float
        Risk-free rate used for Jensen's Alpha computation.
    window_alpha : int, optional (default=25)
        The rolling window size in days used to compute Jensen's Alpha.
    window_beta : int, optional (default=25)
        The rolling window size in days used to compute Beta.
    lookback_steps : list, optional (default=[0, 2, 4, 6, 8, 10])
        Lookback periods used for computing performance metrics.
    show : bool, optional (default=True)
        If True, displays the plot. If False, saves the figure as a temporary PNG file and returns its path.

    Returns:
    -------
    None or str
        If show=True, the function displays the plot. If show=False, it saves the figure and returns the file path.

    Description:
    -----------
    - Computes Jensen's Alpha and Beta for each asset using historical data and varying lookback periods.
    - Each asset is represented by a curve in the Alpha-Beta space, with the initial lookback point marked.
    - Adds vertical and horizontal reference lines for Beta = 1 and Alpha = 0.
    - Uses distinct colors for each asset and provides a legend with lookback period information.
    """
    # Définition des paramètres
    lookback_steps = np.arange(1, 12, 2)
    colors = {s: plt.get_cmap("tab20")(i % 20) for i, s in enumerate(history.columns)}

    # Création des DataFrames pour Jensen's Alpha et Beta
    alpha_df = pd.DataFrame(index=history.columns, columns=lookback_steps+1)
    beta_df = pd.DataFrame(index=history.columns, columns=lookback_steps+1)
    
    for ticker in history:
        for lookback in lookback_steps:
            lookback += 1
            hist_ticker_alpha = history[ticker].iloc[-window_alpha - lookback:-lookback]
            hist_ticker_beta = history[ticker].iloc[-window_beta - lookback:-lookback]
            hist_market_alpha = market.iloc[-window_alpha - lookback:-lookback]
            hist_market_beta = market.iloc[-window_beta - lookback:-lookback]
            
            alpha_df.at[ticker, lookback] = alpha_jensen(hist_ticker_alpha, hist_market_alpha, risk_free) * 100
            beta_df.at[ticker, lookback] = beta(hist_ticker_beta, hist_market_beta, way='all')

    # Tracé du graphique
    fig, ax = plt.subplots(figsize=(8, 8))

    scatter_handles = []
    for ticker in history:
        y_vals = alpha_df.loc[ticker].values
        x_vals = beta_df.loc[ticker].values
        color = colors[ticker]
        
        ax.plot(x_vals, y_vals, color=color, linewidth=1.5, alpha=0.75)
        scatter = ax.scatter(x_vals[0], y_vals[0], color=color, marker="o", s=20)
        scatter_handles.append((scatter, ticker))

    ax.axhline(0, color="grey", linewidth=1, alpha=0.5)
    ax.axvline(1, color="grey", linewidth=1, alpha=0.5)

    # Ajout d'une légende avec points au lieu de lignes
    handles, labels = zip(*scatter_handles)

    # Ajout d'une ligne noire pour la description des lookbacks
    lookback_label = f"Lookback Periods: {lookback_steps[0]} to {lookback_steps[-1]} (Step: {int((lookback_steps.max())/len(lookback_steps))})"
    handles += (plt.Line2D([0], [0], color='black', lw=1),)
    labels += (lookback_label,)

    ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(1, 1))

    plt.title("Relative Performance for Different Tickers")
    plt.ylabel(f"Jensen's Alpha {window_alpha} Days")
    plt.xlabel(f"Beta {window_beta} Days")
    plt.grid(True, alpha=0.5, linestyle="--")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
    
    if show:
        plt.show()
    else:
        rp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(rp_file.name, format='png', bbox_inches='tight')
        plt.close()
        return rp_file
