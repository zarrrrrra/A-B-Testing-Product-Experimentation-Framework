"""
Experiment assignment: deterministic, stateless, auditable.

Each developer is assigned by hashing their id with an experiment-specific
salt. Properties this buys:

  * Idempotent — the same developer always lands in the same arm; no lookup
    table to keep in sync.
  * Stateless — any service (onboarding UI, email, analytics) computes the
    same assignment independently.
  * Isolated — a new salt per experiment yields an independent split, so
    users aren't correlated across experiments.

Also home to the two checks that must pass before a readout is trusted:
sample-ratio mismatch (SRM) and covariate balance.

Run a demo against the profile pool:
    python -m src.experiment --profiles data/profiles.csv
"""

from __future__ import annotations

import argparse
import hashlib

import pandas as pd
from scipy import stats

DEFAULT_SALT = "autoscaling_tip_onboarding_v1"
DEFAULT_TREATMENT_SHARE = 0.50

BALANCE_COVARIATES = ["account_age_days", "repo_count", "commit_streak",
                      "pre_experiment_score"]


def assign_arm(developer_id: str, salt: str = DEFAULT_SALT,
               treatment_share: float = DEFAULT_TREATMENT_SHARE) -> str:
    """Deterministically map a developer id to 'control' or 'treatment'."""
    bucket = int(hashlib.md5(f"{salt}:{developer_id}".encode()).hexdigest()[:8], 16)
    return "treatment" if bucket / 0xFFFFFFFF < treatment_share else "control"


def assign_dataframe(df: pd.DataFrame, id_col: str = "developer_id",
                     salt: str = DEFAULT_SALT,
                     treatment_share: float = DEFAULT_TREATMENT_SHARE,
                     arm_col: str = "arm") -> pd.DataFrame:
    """Return a copy of `df` with an assignment column added."""
    out = df.copy()
    out[arm_col] = out[id_col].astype(str).map(
        lambda d: assign_arm(d, salt, treatment_share))
    return out


def srm_check(df: pd.DataFrame, arm_col: str = "arm",
              treatment_share: float = DEFAULT_TREATMENT_SHARE) -> float:
    """Chi-square sample-ratio-mismatch check. Returns the p-value.

    A tiny p-value (< 0.001) means the observed split is inconsistent with
    the intended allocation — a bug, not bad luck. Do not read results
    until it's explained.
    """
    n_t = int((df[arm_col] == "treatment").sum())
    n_c = int((df[arm_col] == "control").sum())
    total = n_t + n_c
    expected = [total * (1 - treatment_share), total * treatment_share]
    return float(stats.chisquare([n_c, n_t], f_exp=expected).pvalue)


def balance_check(df: pd.DataFrame, covariates: list[str] | None = None,
                  arm_col: str = "arm") -> pd.DataFrame:
    """Welch t-test per covariate across arms.

    With valid randomization ~1 in 20 covariates fails at p < 0.05 by
    chance; a pattern of failures (or failure alongside SRM) is the signal.
    """
    if covariates is None:
        covariates = [c for c in BALANCE_COVARIATES if c in df.columns]
    control = df[df[arm_col] == "control"]
    treatment = df[df[arm_col] == "treatment"]
    rows = []
    for cov in covariates:
        c, t = control[cov].dropna(), treatment[cov].dropna()
        p = float(stats.ttest_ind(c, t, equal_var=False).pvalue)
        rows.append({
            "covariate": cov,
            "control_mean": c.mean(),
            "treatment_mean": t.mean(),
            "difference": t.mean() - c.mean(),
            "p_value": p,
            "balanced": p > 0.05,
        })
    return pd.DataFrame(rows)


def print_assignment_summary(df: pd.DataFrame, arm_col: str = "arm",
                             treatment_share: float = DEFAULT_TREATMENT_SHARE) -> None:
    n_c = int((df[arm_col] == "control").sum())
    n_t = int((df[arm_col] == "treatment").sum())
    print("Assignment summary")
    print("-" * 45)
    print(f"  Control   : {n_c:,}")
    print(f"  Treatment : {n_t:,}")
    print(f"  Treatment share: {n_t / (n_c + n_t):.1%}")

    p = srm_check(df, arm_col, treatment_share)
    print(f"\nSRM check (chi-square vs intended split): p = {p:.4f}")
    if p > 0.001:
        print("  OK -- split is consistent with the intended allocation.")
    else:
        print("  !! SRM DETECTED -- do not read results; find the assignment bug first.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default="data/profiles.csv")
    parser.add_argument("--salt", default=DEFAULT_SALT)
    parser.add_argument("--treatment-share", type=float,
                        default=DEFAULT_TREATMENT_SHARE)
    args = parser.parse_args()

    profiles = pd.read_csv(args.profiles)
    if "developer_id" not in profiles.columns:
        profiles["developer_id"] = profiles.index.astype(str)

    assigned = assign_dataframe(profiles, salt=args.salt,
                                treatment_share=args.treatment_share)
    print_assignment_summary(assigned, treatment_share=args.treatment_share)

    print()
    print("Balance check (we want all of these balanced)")
    print("-" * 45)
    print(balance_check(assigned).to_string(index=False))


if __name__ == "__main__":
    main()
