#!/usr/bin/env python3
"""Generate efficient frontier statistics and visualization for temp.csv data."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS_PER_YEAR = 252


@dataclass
class FrontierResults:
    expected_returns: pd.Series
    volatilities: pd.Series
    covariance: pd.DataFrame
    gmv_weights: pd.Series
    gmv_return: float
    gmv_volatility: float
    max_sharpe_weights: pd.Series | None
    max_sharpe_return: float | None
    max_sharpe_volatility: float | None
    max_sharpe_ratio: float | None
    frontier_points: pd.DataFrame


def load_close_prices(csv_path: Path) -> pd.DataFrame:
    """Read the multi-index CSV and return a Close-price DataFrame indexed by date."""
    df = pd.read_csv(csv_path, header=[0, 1, 2])
    if ('Price', 'Ticker', 'Date') not in df.columns:
        raise ValueError("Expected a ('Price','Ticker','Date') column in the header")

    close = df['Close'].copy()
    close.columns = close.columns.get_level_values(0)
    close.index = pd.to_datetime(df[('Price', 'Ticker', 'Date')])
    close.index.name = 'Date'
    close = close.sort_index()
    return close


def compute_annualized_stats(close_prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """Compute daily returns and annualized statistics from close prices."""
    daily_returns = close_prices.pct_change(fill_method=None).dropna(how='any')
    mu = daily_returns.mean() * TRADING_DAYS_PER_YEAR
    cov = daily_returns.cov() * TRADING_DAYS_PER_YEAR
    return mu, cov, daily_returns


def _solve_frontier(
    mu: pd.Series,
    cov: pd.DataFrame,
    *,
    num_points: int = 75,
    allow_short: bool = False,
    risk_free_rate: float = 0.0,
) -> FrontierResults:
    tickers = list(mu.index)
    n = len(tickers)
    cov_matrix = cov.values
    returns_vec = mu.values

    if allow_short:
        bounds = tuple((None, None) for _ in range(n))
    else:
        bounds = tuple((0.0, 1.0) for _ in range(n))

    def variance(weights: np.ndarray) -> float:
        return float(weights.T @ cov_matrix @ weights)

    def constraint_sum(weights: np.ndarray) -> float:
        return float(np.sum(weights) - 1.0)

    w0 = np.repeat(1.0 / n, n)
    sum_constraint = {'type': 'eq', 'fun': constraint_sum}

    gmv = minimize(variance, w0, method='SLSQP', bounds=bounds, constraints=(sum_constraint,))
    if not gmv.success:
        raise RuntimeError(f"Global minimum variance optimisation failed: {gmv.message}")
    gmv_weights = pd.Series(gmv.x, index=tickers)
    gmv_return = float(np.dot(gmv_weights, returns_vec))
    gmv_vol = float(np.sqrt(variance(gmv_weights.values)))

    target_returns = np.linspace(returns_vec.min(), returns_vec.max(), num_points)
    current_guess = gmv.x
    records = []

    for target in target_returns:
        def constraint_return(weights: np.ndarray, target_return: float = target) -> float:
            return float(np.dot(weights, returns_vec) - target_return)

        constraints = (sum_constraint, {'type': 'eq', 'fun': constraint_return})
        opt = minimize(variance, current_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        if opt.success:
            weights = opt.x
            current_guess = weights
            portfolio_return = float(np.dot(weights, returns_vec))
            portfolio_vol = float(np.sqrt(variance(weights)))
            records.append({'volatility': portfolio_vol, 'return': portfolio_return, **{t: w for t, w in zip(tickers, weights)}})

    frontier_df = pd.DataFrame.from_records(records)

    max_sharpe_weights = max_sharpe_return = max_sharpe_volatility = max_sharpe_ratio = None

    def neg_sharpe(weights: np.ndarray) -> float:
        port_return = float(np.dot(weights, returns_vec))
        port_vol = float(np.sqrt(variance(weights)))
        if port_vol == 0:
            return np.inf
        return -((port_return - risk_free_rate) / port_vol)

    sharpe_opt = minimize(neg_sharpe, gmv.x, method='SLSQP', bounds=bounds, constraints=(sum_constraint,))
    if sharpe_opt.success:
        weights = pd.Series(sharpe_opt.x, index=tickers)
        port_return = float(np.dot(weights, returns_vec))
        port_vol = float(np.sqrt(variance(weights.values)))
        ratio = (port_return - risk_free_rate) / port_vol if port_vol else None
        max_sharpe_weights = weights
        max_sharpe_return = port_return
        max_sharpe_volatility = port_vol
        max_sharpe_ratio = ratio

    return FrontierResults(
        expected_returns=mu,
        volatilities=pd.Series(np.sqrt(np.diag(cov)), index=tickers),
        covariance=cov,
        gmv_weights=gmv_weights,
        gmv_return=gmv_return,
        gmv_volatility=gmv_vol,
        max_sharpe_weights=max_sharpe_weights,
        max_sharpe_return=max_sharpe_return,
        max_sharpe_volatility=max_sharpe_volatility,
        max_sharpe_ratio=max_sharpe_ratio,
        frontier_points=frontier_df,
    )


def generate_plot(frontier: FrontierResults, output_path: Path, risk_free_rate: float = 0.0) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    if not frontier.frontier_points.empty:
        ax.plot(frontier.frontier_points['volatility'], frontier.frontier_points['return'], '-', color='tab:blue', label='Efficient frontier')

    ax.scatter(frontier.volatilities, frontier.expected_returns, marker='o', color='tab:orange', label='Individual assets')
    for ticker, x, y in zip(frontier.expected_returns.index, frontier.volatilities, frontier.expected_returns):
        ax.annotate(ticker, (x, y), textcoords='offset points', xytext=(5, 5))

    ax.scatter([frontier.gmv_volatility], [frontier.gmv_return], marker='D', color='tab:green', label='Global minimum variance')

    if frontier.max_sharpe_weights is not None:
        ax.scatter([frontier.max_sharpe_volatility], [frontier.max_sharpe_return], marker='^', color='tab:red', label='Max Sharpe')
        if frontier.max_sharpe_ratio is not None and risk_free_rate is not None:
            ax.annotate(f"Sharpe {frontier.max_sharpe_ratio:.2f}", (frontier.max_sharpe_volatility, frontier.max_sharpe_return), textcoords='offset points', xytext=(5, -15))

    ax.set_xlabel('Annualized volatility')
    ax.set_ylabel('Annualized return')
    ax.set_title('Efficient Frontier (long-only)')
    ax.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.6)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def format_weights(weights: pd.Series) -> str:
    return ', '.join(f"{ticker}: {weight:.2%}" for ticker, weight in weights.items())


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, default=Path('temp.csv'), help='Path to the input CSV file.')
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('figures/efficient_frontier.svg'),
        help='Where to save the efficient frontier plot. The format is inferred from the file suffix.',
    )
    parser.add_argument('--allow-short', action='store_true', help='Allow short positions when solving the frontier.')
    parser.add_argument('--risk-free-rate', type=float, default=0.0, help='Annualized risk-free rate for the Sharpe ratio (default: 0).')
    parser.add_argument('--points', type=int, default=75, help='Number of points to sample on the frontier curve.')
    args = parser.parse_args(list(argv) if argv is not None else None)

    close_prices = load_close_prices(args.csv)
    mu, cov, _ = compute_annualized_stats(close_prices)
    results = _solve_frontier(mu, cov, num_points=args.points, allow_short=args.allow_short, risk_free_rate=args.risk_free_rate)

    generate_plot(results, args.output, risk_free_rate=args.risk_free_rate)

    print('Annualized expected returns:')
    print((results.expected_returns).apply(lambda x: f"{x:.2%}"))
    print('\nAnnualized volatilities:')
    print((results.volatilities).apply(lambda x: f"{x:.2%}"))
    print('\nGlobal minimum-variance portfolio:')
    print(format_weights(results.gmv_weights))
    print(f"Expected return: {results.gmv_return:.2%}")
    print(f"Volatility: {results.gmv_volatility:.2%}")

    if results.max_sharpe_weights is not None and results.max_sharpe_ratio is not None:
        print('\nMaximum Sharpe portfolio:')
        print(format_weights(results.max_sharpe_weights))
        print(f"Expected return: {results.max_sharpe_return:.2%}")
        print(f"Volatility: {results.max_sharpe_volatility:.2%}")
        print(f"Sharpe ratio (risk-free {args.risk_free_rate:.2%}): {results.max_sharpe_ratio:.2f}")


if __name__ == '__main__':
    main()
