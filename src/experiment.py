"""
Experiment assignment

Assignment by hashing the developer id, plus the sampel ratio mismatch SRM & balance check

Hashing so same developer always lands in the same arm, and any
service can work it out without a lookup table
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
    """deterministically map a developer id to control / treatment"""
    bucket = int(hashlib.md5(f"{salt}:{developer_id}".encode()).hexdigest()[:8], 16)
    return "treatment" if bucket / 0xFFFFFFFF < treatment_share else "control"


def assign_dataframe(df: pd.DataFrame, id_col: str = "developer_id",
                     salt: str = DEFAULT_SALT,
                     treatment_share: float = DEFAULT_TREATMENT_SHARE,
                     arm_col: str = "arm") -> pd.DataFrame:
    out = df.copy()
    out[arm_col] = out[id_col].astype(str).map(
        lambda d: assign_arm(d, salt, treatment_share))
    return out


def srm_check(df: pd.DataFrame, arm_col: str = "arm",
              treatment_share: float = DEFAULT_TREATMENT_SHARE) -> float:
    """Chi-square check on the split, tiny p-value means a bug """

    n_t = int((df[arm_col] == "treatment").sum())
    n_c = int((df[arm_col] == "control").sum())
    total = n_t + n_c
    expected = [total * (1 - treatment_share), total * treatment_share]
    return float(stats.chisquare([n_c, n_t], f_exp=expected).pvalue)


def balance_check(df: pd.DataFrame, covariates: list[str] | None = None,
                  arm_col: str = "arm") -> pd.DataFrame:
    """Welch t test / covariate
     ~ 1 / 20 fails by chance, so look for a pattern rather than a single failure """
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
    print(f"  Control   : {n_c:,}")
    print(f"  Treatment : {n_t:,}")
    print(f"  Treatment share: {n_t / (n_c + n_t):.1%}")


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
    print("Balance check")
    print(balance_check(assigned).to_string(index=False))


if __name__ == "__main__":
    main()
