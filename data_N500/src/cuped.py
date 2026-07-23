"""
CUPED: Controlled-experiment Using Pre-Experiment Data.

Removes the outcome variance that pre-experiment behavior already explains:

    Y_cuped = Y - theta * (X - mean(X)),   theta = cov(Y, X) / var(X)

Variance removed ~= corr(X, Y)^2, so the covariate's predictive power is
everything. The covariate MUST be measured strictly before assignment —
see dbt/models/marts/pre_experiment_covariates.sql for the leakage guard.

Run the comparison on a cohort:
    python -m src.cuped --cohort data/cohort.csv
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.stats import DEFAULT_ALPHA, mean_diff_readout, normalize_cohort


def estimate_theta(y: pd.Series, x: pd.Series) -> float:
    """theta = cov(Y, X) / var(X), estimated on the pooled sample."""
    y, x = pd.Series(y).astype(float), pd.Series(x).astype(float)
    return float(np.cov(y, x, ddof=1)[0, 1] / x.var(ddof=1))


def adjust(y: pd.Series, x: pd.Series, theta: float | None = None) -> pd.Series:
    """Return the CUPED-adjusted outcome."""
    y, x = pd.Series(y).astype(float), pd.Series(x).astype(float)
    if theta is None:
        theta = estimate_theta(y, x)
    return y - theta * (x - x.mean())


def cuped_readout(df: pd.DataFrame, outcome_col: str = "retained",
                  covariate_col: str = "pre_experiment_score",
                  arm_col: str = "arm",
                  alpha: float = DEFAULT_ALPHA) -> dict:
    """Naive vs CUPED-adjusted readout plus adjustment diagnostics."""
    y = df[outcome_col].astype(float)
    x = df[covariate_col].astype(float)

    theta = estimate_theta(y, x)
    y_adj = adjust(y, x, theta)

    is_t = df[arm_col] == "treatment"
    naive = mean_diff_readout(y[is_t], y[~is_t], alpha)
    adjusted = mean_diff_readout(y_adj[is_t], y_adj[~is_t], alpha)

    return {
        "correlation": float(np.corrcoef(x, y)[0, 1]),
        "variance_removed": float(1 - y_adj.var(ddof=1) / y.var(ddof=1)),
        "theta": theta,
        "naive": naive,
        "adjusted": adjusted,
        "ci_narrowing": 1 - adjusted["ci_width_pp"] / naive["ci_width_pp"],
        "adjusted_outcome": y_adj,
    }


def print_cuped(result: dict) -> None:
    print("Does our covariate actually predict the outcome?")
    print("=" * 58)
    print(f"  Correlation (score vs retained) : {result['correlation']:.3f}")
    print(f"  Variance removed by CUPED       : {result['variance_removed']:.1%}")
    print(f"  Theta (adjustment coefficient)  : {result['theta']:.3f}")

    table = pd.DataFrame(
        [result["naive"], result["adjusted"]],
        index=["Naive (unadjusted)", "CUPED-adjusted"],
    ).round(4)
    print()
    print("Naive vs CUPED-adjusted")
    print("=" * 58)
    print(table.to_string())
    print()
    print(f"  Confidence interval narrowed by {result['ci_narrowing']:.1%}")
    print("  (Same effect, measured more precisely -- that's the whole point.)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="data/cohort.csv")
    parser.add_argument("--outcome", default="retained")
    parser.add_argument("--covariate", default="pre_experiment_score")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()

    cohort = normalize_cohort(pd.read_csv(args.cohort))
    if args.covariate not in cohort.columns:
        raise SystemExit(
            f"Cohort has no '{args.covariate}' column. Build "
            "dbt/models/marts/pre_experiment_covariates.sql (or rerun "
            "data/simulate_cohort.py) and join it in."
        )
    print_cuped(cuped_readout(cohort, args.outcome, args.covariate,
                              alpha=args.alpha))


if __name__ == "__main__":
    main()
