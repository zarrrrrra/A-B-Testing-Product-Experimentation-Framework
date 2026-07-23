"""Build the experiment cohort from the real GitHub profiles N=500

Profiles are real, but the retention outcomes are made up.

Output:
    - more active developers (older accounts, more repos, more recent activity) are more likely to retain, regardless of what we show them
    - being in the treatment group (seeing the autoscaling tip) adds a small, realistic lift on top of that (2pp)
    - pre_experiment_score column is produced, which is a measure of how active a developer was, CUPED module uses this to reduce noise
 
"""

import os
import argparse
import numpy as np
import pandas as pd


# The "true" effect of the autoscaling tip, in percentage points of retention.
# This is the ground truth we're pretending not to know when we run the stats.
TRUE_TREATMENT_EFFECT = 0.02  # +2pp
DEFAULT_TRUE_EFFECT_PP = 2.0

# Baseline retention for an average developer with no tip.
BASELINE_RETENTION = 0.61

# How strongly prior engagement predicts retention.
# This is what gives the CUPED covariate something to work with if engagement barely predicts retention, there is no noise for CUPED to remove
ENGAGEMENT_MULTIPLIER = 0.60

RANDOM_SEED = 42

def assign_archetype(row):
    """Put each developer in a bucket so the dashboard is easier to read."""
    age_years = row["account_age_days"] / 365
    if age_years >= 4 and row["repo_count"] >= 30:
        return "experienced"
    if row["commit_streak"] >= 30:
        return "active_hobbyist"
    if age_years < 1:
        return "newcomer"
    return "early_career"


def pre_experiment_score(df):
    """0-1 activity score. Used as the CUPED covariate later."""

    def norm(series):
        # Simple min-max normalization, guarding against divide-by-zero.
        rng = series.max() - series.min()
        if rng == 0:
            return pd.Series(0.5, index=series.index)
        return (series - series.min()) / rng

    age = norm(df["account_age_days"])
    repos = norm(df["repo_count"].clip(upper=200))       # cap outliers
    activity = norm(df["commit_streak"].clip(upper=100))  # cap outliers

    # Weight recent activity most heavily, then repos, then account age.
    score = 0.5 * activity + 0.3 * repos + 0.2 * age
    return score


def simulate_outcomes(profiles: pd.DataFrame,
                      true_effect_pp: float = DEFAULT_TRUE_EFFECT_PP,
                      seed: int = RANDOM_SEED,
                      verbose: bool = True) -> pd.DataFrame:
    """Return a cohort dataframe: profiles + archetype, score, arm, retained."""
    rng = np.random.default_rng(seed)
    cohort = normalize_profiles(profiles, verbose=verbose)

    if "archetype" not in cohort.columns:
        cohort["archetype"] = derive_archetype(cohort)
    cohort["pre_experiment_score"] = compute_pre_experiment_score(cohort, rng)
    cohort["arm"] = cohort["developer_id"].astype(str).map(assign_arm)

    p = (
        cohort["archetype"].map(BASE_RETENTION)
        + SCORE_SLOPE * (cohort["pre_experiment_score"] - 0.25)
        + np.where(cohort["arm"] == "treatment", true_effect_pp / 100, 0.0)
    ).clip(0.02, 0.98)

    cohort["retained"] = (rng.random(len(cohort)) < p).astype(int)
    return cohort

def simulate_cohort(df, seed=RANDOM_SEED):
    """
        archetype             : readable developer bucket
        pre_experiment_score  : 0-1 activity score (CUPED covariate)
        group                 : 'control' or 'treatment'
        saw_autoscaling_tip   : 1 if treatment, else 0
        retained              : 1 if still active after 30 days, else 0
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    df["archetype"] = df.apply(assign_archetype, axis=1)
    df["pre_experiment_score"] = pre_experiment_score(df)

    # 50/50 random assignment 
    df["group"] = rng.choice(["control", "treatment"], size=len(df))
    df["saw_autoscaling_tip"] = (df["group"] == "treatment").astype(int)
   
    # Centering on the cohort's own mean (rather than a hardcoded 0.5) keeps the  AVERAGE developer at BASELINE_RETENTION, so the stated baseline is honest.
    score = df["pre_experiment_score"]
    engagement_shift = (score - score.mean()) * ENGAGEMENT_MULTIPLIER
    prob = BASELINE_RETENTION + engagement_shift
    prob += df["saw_autoscaling_tip"] * TRUE_TREATMENT_EFFECT
    prob = prob.clip(0.01, 0.99)

    df["retained"] = (rng.random(len(df)) < prob).astype(int)

    return df


def summarize(df):
    """sanity check summary of simulated cohort"""
    control = df[df["group"] == "control"]["retained"].mean()
    treatment = df[df["group"] == "treatment"]["retained"].mean()
    print("\nsimulated cohort summary")
    print(f"  Total developers   : {len(df):,}")
    print(f"  Control retention  : {control:.1%}")
    print(f"  Treatment retention: {treatment:.1%}")
    print(f"  Observed lift      : {(treatment - control) * 100:+.1f}pp")
    print(f"  (True effect baked in: {TRUE_TREATMENT_EFFECT * 100:+.1f}pp)")
    print("\n  Archetype breakdown:")
    for name, count in df["archetype"].value_counts().items():
        print(f"    {name:<16} {count:,}")


def main():
    parser = argparse.ArgumentParser(description="Simulate the experiment cohort.")
    parser.add_argument("--in", dest="infile", type=str, default="data_N500/data/profiles.csv",
                        help="Input CSV of real profiles from github_fetch.py")
    parser.add_argument("--out", type=str, default="data_N500/data/cohort.csv",
                        help="Output CSV path for the simulated cohort")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    if not os.path.exists(args.infile):
        raise FileNotFoundError(f"Could not find file")

    profiles = pd.read_csv(args.infile)

    cohort = simulate_cohort(profiles, seed=args.seed)
    summarize(cohort)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cohort.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
